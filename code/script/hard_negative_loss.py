"""Contrastive loss with one curated negative only for its own query.

Unlike broadcasting every mined product to every query, this never assigns an
unverified cross-query negative label to a mined product. Existing in-batch
positive products remain negatives under the original W08 sampling policy.
"""
from __future__ import annotations


def own_negative_loss(queries, positives, negatives=None, negative_rows=(), scale=20.0):
    import torch
    import torch.nn.functional as F
    if queries.ndim != 2 or positives.shape != queries.shape or queries.shape[0] < 2:
        raise ValueError("matching query/positive matrices with at least two rows required")
    if not torch.isfinite(queries).all() or not torch.isfinite(positives).all():
        raise ValueError("non-finite embeddings")
    if (queries.norm(dim=1) <= 0).any() or (positives.norm(dim=1) <= 0).any():
        raise ValueError("zero embeddings")
    q = F.normalize(queries, dim=1)
    p = F.normalize(positives, dim=1)
    logits = q @ p.T * scale
    if negative_rows:
        if (negatives is None or negatives.shape != (len(negative_rows), queries.shape[1])
                or len(set(negative_rows)) != len(negative_rows)
                or min(negative_rows) < 0 or max(negative_rows) >= len(queries)):
            raise ValueError("invalid row-specific negative alignment")
        if not torch.isfinite(negatives).all() or (negatives.norm(dim=1) <= 0).any():
            raise ValueError("invalid negative embeddings")
        indices = torch.tensor(negative_rows, dtype=torch.long, device=queries.device)
        scores = (q[indices] * F.normalize(negatives, dim=1)).sum(dim=1) * scale
        extra = q.new_full((len(q),), float("-inf")).scatter(0, indices, scores)
        logits = torch.cat([logits, extra[:, None]], dim=1)
    elif negatives is not None and len(negatives):
        raise ValueError("negative vectors have no associated queries")
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))
