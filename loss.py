import torch


def cosine_loss(z: torch.tensor, z_hat: torch.tensor) -> torch.tensor:
    """
    Compute the loss defined in the paper.

    Args:
        z (torch.Tensor): The ground truth representation tensor.
        z_hat (torch.Tensor): The predicted tensor.

    Returns:
        torch.Tensor: Negative average cosine similarity as loss.
    """
    cos_fn = torch.nn.CosineSimilarity(dim=2).to(z.device)
    cos_sim = cos_fn(z, z_hat)
    loss = -torch.mean(cos_sim, dim=0).mean()

    return loss
