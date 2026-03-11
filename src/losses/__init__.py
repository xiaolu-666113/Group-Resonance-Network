"""Loss package for GRN."""

from .classification import ClassificationLoss
from .prototype_loss import PrototypeRegularizationLoss

__all__ = ["ClassificationLoss", "PrototypeRegularizationLoss"]
