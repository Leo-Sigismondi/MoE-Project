import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


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



def plot_class_expert_heatmap(model, dataloader, device, num_classes=10, figsize=(8,6)):
    """
    For each true class c and each expert j, computes
    P(sample of class c was assigned to expert j)
    and plots a heatmap.
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
            # track how many of each class we saw
            class_totals += torch.bincount(y, minlength=num_classes).float()

    # normalize per-class
    probs = (counts / class_totals.unsqueeze(1)).cpu().numpy()

    plt.figure(figsize=figsize)
    sns.heatmap(probs, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Exp{j}" for j in range(E)],
                yticklabels=[str(c) for c in range(num_classes)])
    plt.xlabel("Expert")
    plt.ylabel("True Class")
    plt.title("P(true class → expert)")
    plt.tight_layout()
    plt.show()

def plot_expert_embedding_pca(model, dataloader, device, samples_per_expert=200, figsize=(8,6)):
    """
    Runs val samples through the shared trunk, dispatches to each expert,
    collects up to `samples_per_expert` embeddings per expert,
    then does a 2D PCA and scatter-plots colored by expert id.
    """
    model.eval()
    E = model.num_experts
    # C = model.trunk[-1].out_features if hasattr(model.trunk, "__getitem__") else model.trunk[-1]  # not used
    emb_list, owner_list = [], []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            feats = model.trunk(x)          # [B, feat_dim]
            _, _, D = model(x)              # dispatch mask [B, E]

            for j in range(E):
                idx = (D[:, j] > 0.5).nonzero(as_tuple=True)[0]
                if idx.numel():
                    idx = idx[:samples_per_expert]        # cap per-expert
                    emb = model.experts[j](feats[idx])    # [n, embed_dim]
                    emb_list.append(emb.cpu())
                    owner_list.append(torch.full((emb.size(0),), j, dtype=torch.long))
            # stop once we've got enough
            total = sum(o.numel() for o in owner_list)
            if total >= E * samples_per_expert:
                break

    embs = torch.cat(emb_list, dim=0).numpy()
    owners = torch.cat(owner_list).numpy()

    # PCA to 2D
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embs)

    plt.figure(figsize=figsize)
    for j in range(E):
        mask = (owners == j)
        plt.scatter(proj[mask, 0], proj[mask, 1], label=f"Expert {j}", alpha=0.6)
    plt.legend()
    plt.title("PCA of Expert Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()