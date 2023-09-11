"""Datamodels for saving actions."""
import torch
from pydantic import BaseModel, ConfigDict


class SavedAction(BaseModel):
    """Save actions for ActorCritic module."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_prob: torch.Tensor
    value: torch.Tensor
