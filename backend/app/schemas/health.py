"""Schemas for health, liveness, and database readiness telemetry."""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness health check response payload."""

    status: str = Field(default="ok", description="Service operational status")
    service: str = Field(description="Name of the backend service")
    version: str = Field(description="Current deployed version")
    environment: str = Field(description="Runtime deployment environment")


class ReadinessResponse(BaseModel):
    """Readiness probe response payload verifying database connectivity."""

    status: str = Field(description="Readiness status ('ready' or 'degraded')")
    service: str = Field(description="Name of the backend service")
    database: str = Field(description="Database connectivity status ('connected' or 'disconnected')")
    environment: str = Field(description="Runtime deployment environment")
    detail: Optional[str] = Field(default=None, description="Diagnostic detail")
