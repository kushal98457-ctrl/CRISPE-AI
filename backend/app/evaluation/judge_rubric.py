"""Versioned evaluation rubric and dimension criteria for Model-Assisted Evaluation (Step 21).

Defines the 8 core evaluation dimensions, 0-4 ordinal scoring scale, and explicit
grading criteria used by the LLM-as-a-judge to evaluate generated CRISPE prompts.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ModelJudgeDimension(str, Enum):
    """The eight orthogonal dimensions evaluated by the LLM-as-a-judge."""

    INTENT_PRESERVATION = "intent_preservation"
    CLARITY_AND_ACTIONABILITY = "clarity_and_actionability"
    CONTEXT_ADEQUACY = "context_adequacy"
    REQUIREMENT_AND_CONSTRAINT_COVERAGE = "requirement_and_constraint_coverage"
    SPECIFICATION_AND_OUTPUT_PRECISION = "specification_and_output_precision"
    INTERNAL_CONSISTENCY = "internal_consistency"
    EXAMPLE_USEFULNESS = "example_usefulness"
    AMBIGUITY_HANDLING = "ambiguity_handling"


class RubricScoreLevel(int, Enum):
    """0-4 ordinal scale used for model-assisted evaluation."""

    FUNDAMENTALLY_INADEQUATE = 0  # Missing or fundamentally inadequate; unusable for this dimension
    POOR = 1                      # Major omissions or ambiguities make prompt substantially ineffective
    PARTIAL = 2                   # Some relevant guidance exists, but meaningful gaps remain
    GOOD = 3                      # Requirement communicated clearly with only minor shortcomings
    STRONG = 4                    # Communicated precisely and appropriately without important omissions


class DimensionCriterion(BaseModel):
    """Detailed definition, scoring criteria, and applicability rules for a dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: ModelJudgeDimension = Field(..., description="Unique dimension identifier.")
    name: str = Field(..., description="Human-readable title.")
    description: str = Field(..., description="Core quality assessed by this dimension.")
    weight: float = Field(default=1.0, gt=0.0, description="Relative importance weight.")
    applicability_rules: str = Field(..., description="When this dimension applies vs not_applicable.")
    level_descriptions: Dict[int, str] = Field(
        ...,
        description="Explicit description of performance required for scores 0, 1, 2, 3, and 4.",
    )


class EvaluationRubric(BaseModel):
    """Complete versioned rubric containing criteria for all evaluation dimensions."""

    model_config = ConfigDict(extra="forbid")

    rubric_version: str = Field(default="v1.0", description="Semantic version of this rubric.")
    dimensions: Dict[str, DimensionCriterion] = Field(
        ...,
        description="Criteria mapped by dimension identifier string.",
    )

    def render_rubric_text(self) -> str:
        """Render the rubric into structured Markdown for inclusion in judge instructions."""
        lines: List[str] = [
            f"# CRISPE-AI Prompt Evaluation Rubric (Version {self.rubric_version})",
            "",
            "Score every applicable dimension on a 0-4 scale according to the following explicit criteria:",
            "- 0: Missing or fundamentally inadequate",
            "- 1: Poor (major omissions/ambiguities)",
            "- 2: Partial (some relevant guidance, meaningful gaps)",
            "- 3: Good (clear, only minor shortcomings)",
            "- 4: Strong (precise, appropriate, complete)",
            "",
        ]

        for dim_key, criterion in self.dimensions.items():
            lines.append(f"## Dimension: {criterion.name} (`{dim_key}`)")
            lines.append(f"**Description:** {criterion.description}")
            lines.append(f"**Applicability:** {criterion.applicability_rules}")
            lines.append("**Scoring Guide:**")
            for score in (0, 1, 2, 3, 4):
                lines.append(f"- **Score {score}:** {criterion.level_descriptions.get(score, 'N/A')}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Standard V1.0 Rubric Definition
# =============================================================================

DEFAULT_RUBRIC_V1 = EvaluationRubric(
    rubric_version="v1.0",
    dimensions={
        ModelJudgeDimension.INTENT_PRESERVATION.value: DimensionCriterion(
            dimension=ModelJudgeDimension.INTENT_PRESERVATION,
            name="Intent Preservation",
            description="Does the generated prompt accurately represent the user's intended objective without scope drift or alteration?",
            weight=1.5,
            applicability_rules="Always applicable to all prompt generation tasks.",
            level_descriptions={
                0: "The prompt completely alters or misses the user's primary goal; downstream AI will solve the wrong problem.",
                1: "The prompt addresses a tangentially related task, but distorts the core objective or omits major intent.",
                2: "The prompt captures the broad domain, but drifts in scope or misses a central intended use case.",
                3: "The prompt faithfully captures the user's intent with only minor nuances or secondary aspects missed.",
                4: "The prompt impeccably captures the exact objective, perspective, and intended goal of the user.",
            },
        ),
        ModelJudgeDimension.CLARITY_AND_ACTIONABILITY.value: DimensionCriterion(
            dimension=ModelJudgeDimension.CLARITY_AND_ACTIONABILITY,
            name="Clarity and Actionability",
            description="Can the downstream model clearly understand what concrete steps, decisions, and deliverables are required?",
            weight=1.2,
            applicability_rules="Always applicable to all prompt generation tasks.",
            level_descriptions={
                0: "Instructions are incomprehensible, incoherent, or provide no actionable direction.",
                1: "Instructions are excessively vague, leaving the downstream model to guess what deliverables are expected.",
                2: "Instructions outline general direction, but lack concrete methodology, deliverables, or execution steps.",
                3: "Instructions are clear and actionable; downstream model can execute with minimal ambiguity.",
                4: "Instructions are exceptionally crisp, structured, unambiguous, and immediately executable.",
            },
        ),
        ModelJudgeDimension.CONTEXT_ADEQUACY.value: DimensionCriterion(
            dimension=ModelJudgeDimension.CONTEXT_ADEQUACY,
            name="Context Adequacy",
            description="Does the prompt include necessary operational circumstances without hallucinating unprovided requirements?",
            weight=1.0,
            applicability_rules="Always applicable. Evaluates both presence of supplied context and avoidance of ungrounded assumptions.",
            level_descriptions={
                0: "No context provided, or context introduces fabricated domain requirements contradicting the user request.",
                1: "Context is severely impoverished or contains major unsupported assumptions that alter problem scope.",
                2: "Context is minimal, omitting helpful background provided in the request or making minor unstated leaps.",
                3: "Context adequately conveys the necessary background and environment with well-calibrated assumptions.",
                4: "Context provides a rich, realistic, and faithful operational framing tailored exactly to the task.",
            },
        ),
        ModelJudgeDimension.REQUIREMENT_AND_CONSTRAINT_COVERAGE.value: DimensionCriterion(
            dimension=ModelJudgeDimension.REQUIREMENT_AND_CONSTRAINT_COVERAGE,
            name="Requirement and Constraint Coverage",
            description="Does the prompt faithfully preserve mandatory technical requirements, limitations, and negative boundaries?",
            weight=1.5,
            applicability_rules="Always applicable whenever explicit requirements or constraints are provided.",
            level_descriptions={
                0: "Violates or omits all critical requirements and constraints recorded in the task definition.",
                1: "Omits multiple critical requirements or allows explicit prohibited behaviors.",
                2: "Preserves some requirements, but omits at least one critical constraint or key technical requirement.",
                3: "Preserves all critical requirements and most minor constraints with only negligible omissions.",
                4: "Flawlessly preserves every critical requirement, constraint, technology lock, and negative boundary.",
            },
        ),
        ModelJudgeDimension.SPECIFICATION_AND_OUTPUT_PRECISION.value: DimensionCriterion(
            dimension=ModelJudgeDimension.SPECIFICATION_AND_OUTPUT_PRECISION,
            name="Specification and Output Precision",
            description="Does the prompt clearly specify output formatting, schema, granularity, and structure?",
            weight=1.0,
            applicability_rules="Always applicable.",
            level_descriptions={
                0: "No formatting or structure specified, leaving downstream format entirely unconstrained.",
                1: "Output specification is confusing, contradictory, or severely underspecified.",
                2: "Specifies a generic format (e.g. 'write text') but ignores requested structured schemas, tables, or fields.",
                3: "Specifies expected structure, format, and sections clearly with minor ambiguities.",
                4: "Provides exact, rigorous schema, structural blueprints, and precise output formatting rules.",
            },
        ),
        ModelJudgeDimension.INTERNAL_CONSISTENCY.value: DimensionCriterion(
            dimension=ModelJudgeDimension.INTERNAL_CONSISTENCY,
            name="Internal Consistency",
            description="Are the instructions across all six CRISPE sections coherent and mutually compatible, without self-contradiction?",
            weight=1.0,
            applicability_rules="Always applicable across the 6 CRISPE sections.",
            level_descriptions={
                0: "Severe internal contradictions render the prompt completely unimplementable (e.g. conflicting technology mandates).",
                1: "Multiple contradictory requirements or conflicting instructions between sections (e.g. instruction contradicts specification).",
                2: "Minor friction or inconsistent terminology between sections, though overall direction is salvageable.",
                3: "Instructions are coherent across sections with only trivial stylistic or semantic friction.",
                4: "Complete harmony and seamless alignment across all six sections; perfectly consistent terminology and mandates.",
            },
        ),
        ModelJudgeDimension.EXAMPLE_USEFULNESS.value: DimensionCriterion(
            dimension=ModelJudgeDimension.EXAMPLE_USEFULNESS,
            name="Example Usefulness",
            description="Where an example is appropriate, does it meaningfully clarify the expected execution pattern rather than add noise?",
            weight=0.8,
            applicability_rules="Applicable to the Example section. If task needs no example, a concise illustrative template is sufficient.",
            level_descriptions={
                0: "Example is completely nonsensical, misleading, or contradicts the instruction.",
                1: "Example is irrelevant, misleading, or demonstrates poor practices that violate specifications.",
                2: "Example is generic placeholder text (e.g. 'foo bar') that adds minimal illustrative value.",
                3: "Example is realistic, concrete, and clarifies the expected output format and style.",
                4: "Example is an exemplary demonstration illustrating tricky edge cases, precise format, and high quality.",
            },
        ),
        ModelJudgeDimension.AMBIGUITY_HANDLING.value: DimensionCriterion(
            dimension=ModelJudgeDimension.AMBIGUITY_HANDLING,
            name="Ambiguity Handling",
            description="Does the prompt handle uncertainties constructively by stating reasonable assumptions or noting clarifications?",
            weight=1.0,
            applicability_rules="Evaluates how the prompt navigates ambiguity. If task was fully specified, score based on not inventing false ambiguity.",
            level_descriptions={
                0: "Prompt makes wild, unwarranted assumptions that derail the task, or fails completely in the face of ambiguity.",
                1: "Ignores major ambiguities noted in the task, making silent, brittle assumptions.",
                2: "Handles some ambiguity, but fails to document critical assumptions needed for predictable execution.",
                3: "Explicitly states sensible default assumptions or requests necessary clarification constructively.",
                4: "Masterfully navigates ambiguity, providing clear contingency branches, explicit assumptions, or targeted questions.",
            },
        ),
    },
)


def get_default_rubric() -> EvaluationRubric:
    """Return the canonical V1.0 evaluation rubric."""
    return DEFAULT_RUBRIC_V1
