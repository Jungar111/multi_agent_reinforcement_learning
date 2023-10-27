"""Classes for logging."""
from pydantic import BaseModel, Field


class ModelLog(BaseModel):
    """Pydantic basemodel to log training/test."""

    reward: float = Field(default=0.0)
    served_demand: int = Field(default=0)
    rebalancing_cost: float = Field(default=0.0)

    def dict(self, name):
        """Return logs as a dict."""
        return {f"{name}_{key}": val for key, val in dict(self).items()}
