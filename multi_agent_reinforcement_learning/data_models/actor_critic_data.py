"""Datamodels for saving actions."""
import torch
from pydantic import BaseModel, ConfigDict
import typing as T


class SavedAction(BaseModel):
    """Save actions for ActorCritic module."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_prob: torch.Tensor
    log_prob_price: T.Optional[torch.Tensor] = None
    value: torch.Tensor
