"""Schemas for standardized error responses."""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Field-specific validation error details."""

    loc: Optional[List[Any]] = Field(default=None, description="Location of the field in request")
    msg: str = Field(description="Error message for the specific field")
    type: str = Field(description="Error type identifier")


class ErrorResponse(BaseModel):
    """Standardized API error response format."""

    status_code: int = Field(description="HTTP status code")
    error_code: str = Field(description="Machine-readable error classification code")
    message: str = Field(description="Human-readable explanation of the error")
    request_id: Optional[str] = Field(default=None, description="Correlation identifier for tracing")
    details: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed field errors if applicable")
