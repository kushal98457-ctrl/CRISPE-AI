"""Schemas for health and liveness telemetry."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness health check response payload."""

    status: str = Field(default="ok", description="Service operational status")
    service: str = Field(description="Name of the backend service")
    version: str = Field(description="Current deployed version")
    environment: str = Field(description="Runtime deployment environment")
