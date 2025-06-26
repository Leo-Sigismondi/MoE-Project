import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
            # Use assignment mask from model with return_aux=True
            _, _, assignment, *_ = model(x, return_aux=True, targets=y)  # assignment: [B, E]
            for j in range(E):
                idx = (assignment[:, j] > 0.5).nonzero(as_tuple=True)[0]
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
        cmap="plasma",
        xticklabels=[f"Exp{j}" for j in range(E)],
        yticklabels=[str(c) for c in range(num_classes)]
    )
    plt.xlabel("Expert")
    plt.ylabel("True Class")
    plt.title("P(true class → expert)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# def plot_expert_embedding_pca(model, dataloader, device, samples_per_expert=200, figsize=(8,6), save_dir="plot", filename="expert_embedding_pca.png"):
#     """
#     Runs val samples through the shared trunk, dispatches to each expert,
#     collects up to `samples_per_expert` embeddings per expert,
#     then does a 2D PCA and saves a scatter-plot image colored by expert id.
#     """
#     model.eval()
#     E = model.num_experts
#     emb_list, owner_list = [], []

#     with torch.no_grad():
#         for x, _ in dataloader:
#             x = x.to(device)
#             feats = model.trunk(x)          # [B, feat_dim]
#             _, _, D = model(x)              # dispatch mask [B, E]

#             for j in range(E):
#                 idx = (D[:, j] > 0.5).nonzero(as_tuple=True)[0]
#                 if idx.numel():
#                     idx = idx[:samples_per_expert]
#                     emb = model.experts[j](feats[idx])
#                     emb_list.append(emb.cpu())
#                     owner_list.append(torch.full((emb.size(0),), j, dtype=torch.long))
#             total = sum(o.numel() for o in owner_list)
#             if total >= E * samples_per_expert:
#                 break

#     embs = torch.cat(emb_list, dim=0).numpy()
#     owners = torch.cat(owner_list).numpy()

#     pca = PCA(n_components=2)
#     proj = pca.fit_transform(embs)

#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure(figsize=figsize)
#     for j in range(E):
#         mask = (owners == j)
#         plt.scatter(proj[mask, 0], proj[mask, 1], label=f"Expert {j}", alpha=0.6)
#     plt.legend()
#     plt.title("PCA of Expert Embeddings")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, filename))
#     plt.close()

# def plot_per_expert_confusion(model, dataloader, device, num_classes=10, save_dir="conf_mtx"):
#     model.eval()
#     E = model.num_experts

#     # store preds & trues per expert
#     preds_per_e = {j: [] for j in range(E)}
#     trues_per_e = {j: [] for j in range(E)}

#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             # get final gated‐ensemble logits and dispatch mask D
#             logits, _, D, *_ = model(x)
#             pred = logits.argmax(dim=1).cpu().numpy()
#             y_cpu = y.cpu().numpy()
 
#             # for each expert, gather samples it handled
#             for j in range(E):
#                 idxs = (D[:,j] > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
#                 if len(idxs)==0:
#                     continue
#                 preds_per_e[j].append(pred[idxs])
#                 trues_per_e[j].append(y_cpu[idxs])

#     os.makedirs(save_dir, exist_ok=True)
#     # now plot one confusion matrix per expert
#     for j in range(E):
#         if not preds_per_e[j]:
#             continue
#         preds = np.concatenate(preds_per_e[j])
#         trues = np.concatenate(trues_per_e[j])
#         cm = confusion_matrix(trues, preds, labels=np.arange(num_classes))

#         plt.figure(figsize=(5,4))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                     xticklabels=[str(i) for i in range(num_classes)],
#                     yticklabels=[str(i) for i in range(num_classes)])
#         plt.title(f"Expert {j} Confusion Matrix")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f"expert_{j}_confusion_matrix.png"))
#         plt.close()



def plot_per_expert_confusion(model, dataloader, device, num_classes=10, save_dir="plots/confusion_matrix"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    all_preds = []
    all_labels = []
    all_experts = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits, probs, assignment, *_ = model(images, return_aux=True, targets=labels)
            preds = logits.argmax(dim=1)
            # assignment: [B, num_experts], bool
            # For each sample, find which expert was assigned (should be only one True per row)
            expert_ids = assignment.float().argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_experts.append(expert_ids.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_experts = torch.cat(all_experts)

    for expert_id in range(model.num_experts):
        idx = (all_experts == expert_id)
        if idx.sum() == 0:
            continue
        cm = confusion_matrix(all_labels[idx], all_preds[idx], labels=range(num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title(f"Confusion Matrix for Expert {expert_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_expert_{expert_id}.png"))
        plt.close()

def plot_expert_embedding_pca(model, dataloader, device, save_dir="plots/expert_embeddings", filename="expert_embeddings_pca.png", figsize=(10, 10)):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    embeddings = []
    experts = []
    labels = []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            lbls = lbls.to(device)
            # Get shared features from the trunk
            shared_features = model.encoder(images)  # (B, c2, H, W)
            flat_features = shared_features.mean(dim=(-2, -1))  # (B, c2)
            logits, probs, assignment, *_ = model(images, return_aux=True, targets=lbls)
            expert_ids = assignment.float().argmax(dim=1)
            for i in range(images.size(0)):
                expert_idx = expert_ids[i].item()
                expert = model.experts[expert_idx]
                # Pass the shared features for this sample to the expert's encoder/project
                expert_input = shared_features[i:i+1]
                emb = expert.encoder(expert_input)
                emb = expert.project(emb)
                emb = emb.cpu().squeeze(0)
                embeddings.append(emb)
                experts.append(expert_idx)
                labels.append(lbls[i].item())

    embeddings = torch.stack(embeddings).numpy()
    experts = np.array(experts)
    labels = np.array(labels)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=experts, cmap='tab10', alpha=0.6, s=10)
    plt.legend(*scatter.legend_elements(), title="Expert")
    plt.title("Expert Embeddings PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()