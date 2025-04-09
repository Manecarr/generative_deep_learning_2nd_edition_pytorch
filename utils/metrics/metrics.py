import torch
from torch import Tensor


def from_logits_to_class_ids(logits: Tensor) -> Tensor:
    """Convert logits to corresponding class ids.

    Args:
        logits: logits tensor of shape (batch_size, num_classes).

    Returns:
        tensor of shape (batch_size, ) with the class ids.
    """
    return torch.argmax(logits, dim=1)


def calculate_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Calculate the accuracy of the model.

    Args:
        logits: logits tensor of shape (batch_size, num_classes).
        labels: tensor of shape (batch_size, ) with the class ids.

    Returns:
        accuracy as a float.
    """
    predictions = from_logits_to_class_ids(logits)
    correct_predictions = torch.sum(predictions == labels).item()
    accuracy: float = correct_predictions / labels.size(0)
    return accuracy
