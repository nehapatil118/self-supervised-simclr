import torch
import torch.nn.functional as F


# NT-Xent (contrastive) loss used in SimCLR
def nt_xent_loss(z1, z2, temperature=0.1):
    batch_size = z1.size(0)  # Batch size

    # Concatenate both views
    z = torch.cat([z1, z2], dim=0)

    # Compute cosine similarity matrix
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    similarity = similarity.masked_fill(mask, -9e15)

    # Extract positive pairs
    positives = torch.cat([
        torch.diag(similarity, batch_size),
        torch.diag(similarity, -batch_size)
    ])

    # Labels: positives at index 0
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

    # Combine positives and negatives into logits
    logits = torch.cat([positives.unsqueeze(1), similarity], dim=1)

    # Compute contrastive loss
    loss = F.cross_entropy(logits / temperature, labels)

    return loss
