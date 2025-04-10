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


def calculate_tp_fp_tn_fn(logits: Tensor, labels: Tensor) -> tuple[int, int, int, int, float, float, float, float]:
    """Calculate the number of true positives, false positives, true negatives and false negatives.

    Results will be reported both in absolute numbers and as a percentage of the total number of samples.

    .. note::
        If there are more than two classes, only TPs and FPs will be calculated.
        The other metrics will be set to 0.

    Args:
        logits: logits tensor of shape (batch_size, num_classes).
        labels: tensor of shape (batch_size, ) with the class ids.

    Returns:
        the metrics.
    """
    predictions = from_logits_to_class_ids(logits)
    num_samples = labels.size(0)
    if labels.unique().size(0) > 2:
        tps = torch.sum(predictions == labels).item()
        fps = torch.sum(predictions != labels).item()
        tns = 0.0
        fns = 0.0
    else:
        tps = torch.sum((predictions == 1) & (labels == 1)).item()
        fps = torch.sum((predictions == 1) & (labels == 0)).item()
        tns = torch.sum((predictions == 0) & (labels == 0)).item()
        fns = torch.sum((predictions == 0) & (labels == 1)).item()
    return (
        int(tps),
        int(fps),
        int(tns),
        int(fns),
        tps / num_samples,
        fps / num_samples,
        tns / num_samples,
        fns / num_samples,
    )
