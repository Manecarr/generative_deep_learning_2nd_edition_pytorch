from collections.abc import Callable

import torch
from torch.distributions import MultivariateNormal, kl_divergence
from torch.nn.modules.loss import _Loss


class Lambda(torch.nn.Module):
    """A simple wrapper for a lambda function to be used as a PyTorch module."""

    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Initialize the module with the given function."""
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the function to the input tensor."""
        return self.func(x)


class KLNormalLoss(_Loss):
    """The KL divergence between a normal distributions and the standard normal."""

    def forward(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Calculate the loss."""
        # Batch size can change in the last epoch. These distributions need to be instantiated dynamically.
        std_normal = MultivariateNormal(torch.zeros_like(mean), scale_tril=torch.diag_embed(torch.ones_like(mean)))
        distr = MultivariateNormal(mean, scale_tril=torch.diag_embed(var + 1e-6))
        loss = kl_divergence(distr, std_normal)
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}.")


class VAELoss(torch.nn.Module):
    """The loss function for the VAE.

    This is a combination of the reconstruction loss, a binary cross entropy, and the KL divergence loss.
    """

    def __init__(self, weights: tuple[float, float]) -> None:
        """Initialize the loss function.

        Args:
            weights: the weigths for the reconstruction and the KL losses, respectively.
        """
        super().__init__()
        self.rec_w, self.kl_w = weights
        self.rec_loss = torch.nn.BCELoss(reduction="none")
        self.kl_loss = KLNormalLoss(reduction="none")

    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the loss."""
        rec_loss = self.rec_loss(preds, gts).mean(axis=(1, 2, 3))
        kl_loss = self.kl_loss(mean, var)
        loss = self.rec_w * rec_loss + self.kl_w * kl_loss
        return rec_loss, kl_loss, loss
