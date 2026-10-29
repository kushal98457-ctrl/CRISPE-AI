"""Pydantic schemas and data contracts for prompt optimization evaluation dataset (Step 18).

Provides typed representations for evaluation examples used by the Self-Evolving
Prompt Engine (DSPy and GEPA optimization workflows).
"""

from enum import Enum
import re
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    """Categorical domain taxonomy for prompt optimization evaluation tasks."""

    SOFTWARE_ENGINEERING = "software_engineering"
    AI_MACHINE_LEARNING = "ai_machine_learning"
    TECHNICAL_RESEARCH = "technical_research"
    EDUCATION_PROBLEM_SOLVING = "education_problem_solving"
    PRODUCT_SYSTEM_DESIGN = "product_system_design"
    EDGE_CASES_CONFLICTS = "edge_cases_conflicts"


class TaskDifficulty(str, Enum):
    """Cognitive and constraint complexity of the evaluation task."""

    EASY = "easy"        # Clear single-objective tasks with straightforward explicit constraints
    MEDIUM = "medium"    # Multi-requirement tasks with technical domain context and structured constraints
    HARD = "hard"        # Complex tasks involving competing requirements, ambiguity, or edge cases


class EvaluationExample(BaseModel):
    """Complete specification of a prompt generation evaluation benchmark record."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    # -------------------------------------------------------------------------
    # Identifier & Taxonomical Metadata
    # -------------------------------------------------------------------------
    example_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Stable, unique alphanumeric identifier for the evaluation example.",
    )
    task_family: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional grouping identifier to keep related tasks together during future dataset partitioning.",
    )
    task_type: TaskType = Field(
        ...,
        description="High-level domain category representing the engineering context.",
    )
    difficulty: TaskDifficulty = Field(
        ...,
        description="Assigned complexity level based on constraint burden and ambiguity handling.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Searchable keyword descriptors characterizing the specific sub-topic or techniques.",
    )
    example_version: str = Field(
        default="v1",
        max_length=16,
        description="Semantic version of this benchmark record for reproducibility.",
    )

    # -------------------------------------------------------------------------
    # Generator Input Fields (Provided to CRISPE Generation Service)
    # -------------------------------------------------------------------------
    user_idea: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Raw task idea or instruction submitted to CRISPE-AI for prompt generation.",
    )
    task_context: str = Field(
        ...,
        min_length=10,
        description="Operational domain circumstances and background required to interpret the idea.",
    )
    required_behaviors: List[str] = Field(
        ...,
        min_length=1,
        description="Observable actions and explicit requirements the generated prompt must instruct the target AI to execute.",
    )
    constraints: List[str] = Field(
        ...,
        min_length=1,
        description="Explicit boundaries, limitations, technology locks, and negative requirements to preserve.",
    )
    expected_output_format: str = Field(
        ...,
        min_length=5,
        description="The structural format that the downstream response must follow (e.g. Markdown schema, JSON, tables).",
    )
    ambiguity_notes: str = Field(
        ...,
        min_length=4,
        description="Specific areas of uncertainty or missing inputs that the prompt must handle (or 'None' if fully specified).",
    )

    # -------------------------------------------------------------------------
    # Evaluator & Ground Truth Guidance Fields (Used to Grade Generated CRISPE Prompt)
    # -------------------------------------------------------------------------
    acceptance_criteria: List[str] = Field(
        ...,
        min_length=1,
        description="Objective checkpoints used by the evaluation engine to assess whether the prompt represents the task.",
    )
    critical_requirements: List[str] = Field(
        ...,
        min_length=1,
        description="Mandatory conditions whose omission by the generated prompt triggers a significant score deduction.",
    )
    prohibited_behaviors: List[str] = Field(
        ...,
        min_length=1,
        description="Antipatterns or failure modes the generated prompt must NOT encourage or permit.",
    )
    grading_notes: str = Field(
        ...,
        min_length=10,
        description="Clear instructions for future automated and human evaluators on applying the rubric consistently.",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("example_id")
    @classmethod
    def validate_example_id_format(cls, value: str) -> str:
        """Enforce standard lowercase slug format for example IDs."""
        cleaned = value.strip().lower()
        if not re.match(r"^[a-z0-9_\-]+$", cleaned):
            raise ValueError(
                f"example_id '{value}' must contain only lowercase letters, digits, underscores, or hyphens."
            )
        return cleaned

    @field_validator("user_idea", "task_context", "expected_output_format", "ambiguity_notes", "grading_notes")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Strip surrounding whitespace from string fields."""
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("String field cannot be empty or whitespace only.")
            return cleaned
        return value

    @field_validator("required_behaviors", "constraints", "acceptance_criteria", "critical_requirements", "prohibited_behaviors")
    @classmethod
    def validate_list_items(cls, items: List[str]) -> List[str]:
        """Ensure list contains non-empty, stripped strings."""
        if not items:
            raise ValueError("Field list cannot be empty.")
        cleaned_list: List[str] = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("List items must be non-empty strings.")
            cleaned_list.append(item.strip())
        return cleaned_list
