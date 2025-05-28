import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np
import os


def compute_importance(scores):
    """
    Per-expert “importance” = average gate probability over the batch.
    scores: [B, E] raw or post-softmax gate scores (should sum to 1 per row).
    returns: [E]
    """
    probs = F.softmax(scores, dim=1)
    return probs.mean(dim=0)


def compute_load(dispatch_mask):
    """
    Per-expert “load” = fraction of samples assigned to each expert.
    dispatch_mask: [B, E] binary (1 if sample→expert, 0 otherwise)
    returns: [E]
    """
    B = dispatch_mask.size(0)
    return dispatch_mask.sum(dim=0).float() / B


def compute_entropy(scores):
    """
    Average per-sample entropy of the gate distribution.
    scores: [B, E] post-softmax
    returns: scalar
    """
    probs = F.softmax(scores, dim=1)
    eps = 1e-8
    ent_per_sample = -(probs * (probs + eps).log()).sum(dim=1)
    return ent_per_sample.mean()


# ——— Additional diagnostics ———

def compute_sample_load(dispatch_mask):
    """
    How many experts each sample got (on average).
    Ideal = your k.
    returns: scalar
    """
    # sum over experts per sample, then average
    return dispatch_mask.sum(dim=1).float().mean()


def compute_expert_utilization(dispatch_mask, capacity):
    """
    For each expert, what fraction of its capacity did it fill?
    returns: [E]
    """
    expert_count = dispatch_mask.sum(dim=0).float()      # how many samples each expert got
    return expert_count / capacity


def compute_load_variance(dispatch_mask):
    """
    Variance of the expert loads—if this is high, some experts are doing 
    way more work than others.
    returns: scalar
    """
    loads = dispatch_mask.sum(dim=0).float()
    return loads.var()


def compute_top1_distribution(scores):
    """
    If you look at each sample’s top-1 expert (highest score),
    what fraction of samples favor each expert?
    scores: [B, E]
    returns: [E]
    """
    # argmax per row → counts
    top1 = scores.argmax(dim=1)
    counts = torch.bincount(top1, minlength=scores.size(1)).float()
    return counts / scores.size(0)


def compute_expert_class_mi_top1(scores, targets):
    """
    Mutual Information I(Expert;Class) where Expert = argmax(scores, dim=1).
    scores:  [B, E]  raw gate logits or probabilities
    targets: [B]     ground-truth class labels in [0..C-1]
    returns: scalar MI >= 0
    """
    B, E = scores.size()
    # 1) pick top1 expert per sample
    top1 = scores.argmax(dim=1)           # [B]

    # 2) joint counts
    C = int(targets.max().item()) + 1
    counts = torch.zeros(E, C, device=scores.device)
    for e in range(E):
        idxs = (top1 == e).nonzero(as_tuple=True)[0]
        if idxs.numel():
            binc = torch.bincount(targets[idxs], minlength=C).float()
            counts[e] = binc

    # 3) normalize to get true joint P(e,c)
    p_e_c = counts / B                    # sums to 1

    # 4) marginals
    p_e = p_e_c.sum(dim=1, keepdim=True)  # [E,1]
    p_c = p_e_c.sum(dim=0, keepdim=True)  # [1,C]

    # 5) clamp to avoid log(0)
    eps = 1e-12
    p_e_c = p_e_c.clamp(min=eps)
    p_e   = p_e.clamp(min=eps)
    p_c   = p_c.clamp(min=eps)

    # 6) compute MI
    mi = (p_e_c * (p_e_c.log() - p_e.log() - p_c.log())).sum()
    return mi



def plot_class_expert_heatmap(model, dataloader, device, num_classes=10, figsize=(8,6), save_dir="plot", filename="class_expert_heatmap.png"):
    """
    For each true class c and each expert j, computes
    P(sample of class c was assigned to expert j)
    and saves a heatmap image to the specified folder.
    """
    model.eval()
    E = model.num_experts
    counts = torch.zeros(num_classes, E, device=device)
    class_totals = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, _, D = model(x)              # D: [B, E] dispatch mask
            for j in range(E):
                idx = (D[:, j] > 0.5).nonzero(as_tuple=True)[0]
                cls = y[idx]
                if cls.numel():
                    binc = torch.bincount(cls, minlength=num_classes).float()
                    counts[:, j] += binc
            class_totals += torch.bincount(y, minlength=num_classes).float()

    # Avoid division by zero
    class_totals = class_totals.clamp(min=1e-8)
    probs = (counts / class_totals.unsqueeze(1)).cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=figsize)
    sns.heatmap(
        probs,
        annot=True,
        fmt=".2f",
        cmap="hot",  # Changed to a more common color map
        xticklabels=[f"Exp{j}" for j in range(E)],
        yticklabels=[str(c) for c in range(num_classes)]
    )
    plt.xlabel("Expert")
    plt.ylabel("True Class")
    plt.title("P(true class → expert)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_expert_embedding_pca(model, dataloader, device, samples_per_expert=200, figsize=(8,6), save_dir="plot", filename="expert_embedding_pca.png"):
    """
    Runs val samples through the shared trunk, dispatches to each expert,
    collects up to `samples_per_expert` embeddings per expert,
    then does a 2D PCA and saves a scatter-plot image colored by expert id.
    """
    model.eval()
    E = model.num_experts
    emb_list, owner_list = [], []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            feats = model.trunk(x)          # [B, feat_dim]
            _, _, D = model(x)              # dispatch mask [B, E]

            for j in range(E):
                idx = (D[:, j] > 0.5).nonzero(as_tuple=True)[0]
                if idx.numel():
                    idx = idx[:samples_per_expert]
                    emb = model.experts[j](feats[idx])
                    emb_list.append(emb.cpu())
                    owner_list.append(torch.full((emb.size(0),), j, dtype=torch.long))
            total = sum(o.numel() for o in owner_list)
            if total >= E * samples_per_expert:
                break

    embs = torch.cat(emb_list, dim=0).numpy()
    owners = torch.cat(owner_list).numpy()

    pca = PCA(n_components=2)
    proj = pca.fit_transform(embs)

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=figsize)
    for j in range(E):
        mask = (owners == j)
        plt.scatter(proj[mask, 0], proj[mask, 1], label=f"Expert {j}", alpha=0.6)
    plt.legend()
    plt.title("PCA of Expert Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_per_expert_confusion(model, dataloader, device, num_classes=10, save_dir="conf_mtx"):
    model.eval()
    E = model.num_experts

    # store preds & trues per expert
    preds_per_e = {j: [] for j in range(E)}
    trues_per_e = {j: [] for j in range(E)}

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # get final gated‐ensemble logits and dispatch mask D
            logits, conf, D = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_cpu = y.cpu().numpy()

            # for each expert, gather samples it handled
            for j in range(E):
                idxs = (D[:,j] > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
                if len(idxs)==0:
                    continue
                preds_per_e[j].append(pred[idxs])
                trues_per_e[j].append(y_cpu[idxs])

    os.makedirs(save_dir, exist_ok=True)
    # now plot one confusion matrix per expert
    for j in range(E):
        if not preds_per_e[j]:
            continue
        preds = np.concatenate(preds_per_e[j])
        trues = np.concatenate(trues_per_e[j])
        cm = confusion_matrix(trues, preds, labels=np.arange(num_classes))

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[str(i) for i in range(num_classes)],
                    yticklabels=[str(i) for i in range(num_classes)])
        plt.title(f"Expert {j} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"expert_{j}_confusion_matrix.png"))
        plt.close()