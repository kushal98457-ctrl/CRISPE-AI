"""Database repositories for CRISPE-AI."""

from app.repositories.principal_repository import PrincipalRepository
from app.repositories.prompt_repository import PromptRepository

__all__ = [
    "PrincipalRepository",
    "PromptRepository",
]
