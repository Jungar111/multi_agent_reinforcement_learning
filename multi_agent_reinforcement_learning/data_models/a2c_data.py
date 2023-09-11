"""Datamodels for saving actions."""
from pydantic import BaseModel


class SavedAction(BaseModel):
    """Save actions for ActorCritic module."""

    log_prob: float
    value: float
