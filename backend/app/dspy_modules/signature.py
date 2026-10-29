"""DSPy Signature for canonical CRISPE prompt generation (Step 22).

Defines the typed input and output contracts for transforming user ideas and
operational task context into structured, production-grade CRISPE framework prompts.
"""

from typing import Type
import dspy


class CrispeGenerationSignature(dspy.Signature):
    """Transform a user's prompt idea, domain context, and constraints into a production-grade CRISPE prompt.

    The generated output must strictly populate the six canonical CRISPE framework sections:
    - Context: The operational background, domain circumstances, and environment.
    - Role: The persona, authority, perspective, and professional identity.
    - Instruction: The primary task, explicit action commands, and step-by-step methodologies.
    - Specification: Constraints, formatting rules, length guidelines, and negative boundaries.
    - Performance: Evaluation criteria, success benchmarks, and qualitative standards.
    - Example: Concrete, illustrative sample input/output demonstrating the desired execution pattern.
    """

    # -------------------------------------------------------------------------
    # Generator Input Fields
    # -------------------------------------------------------------------------
    idea: str = dspy.InputField(
        desc="The core task, raw prompt idea, or initial instruction to transform into the CRISPE framework."
    )
    task_context: str = dspy.InputField(
        default="",
        desc="Operational domain circumstances, user environment, and background necessary to interpret the task."
    )
    constraints: str = dspy.InputField(
        default="",
        desc="Explicit technical boundaries, limitations, technology locks, and negative rules to preserve."
    )
    output_format_preference: str = dspy.InputField(
        default="",
        desc="Structural format or schema desired for the downstream response (e.g., Markdown, JSON, table)."
    )

    # -------------------------------------------------------------------------
    # Canonical CRISPE Output Fields (Aligns with app.schemas.crispe.CrispeSections)
    # -------------------------------------------------------------------------
    context: str = dspy.OutputField(
        desc="C - Context: The background situation, domain circumstances, operational setting, and business context."
    )
    role: str = dspy.OutputField(
        desc="R - Role: The persona, authority, perspective, professional identity, and tone of voice the AI must embody."
    )
    instruction: str = dspy.OutputField(
        desc="I - Instruction: The primary task, explicit actions, step-by-step commands, and direct methodologies."
    )
    specification: str = dspy.OutputField(
        desc="S - Specification: Constraints, formatting rules, length guidelines, structural requirements, and negative boundaries."
    )
    performance: str = dspy.OutputField(
        desc="P - Performance: Evaluation criteria, success benchmarks, qualitative standards, and output quality metrics."
    )
    example: str = dspy.OutputField(
        desc="E - Example: Concrete, illustrative sample input/output demonstrations or reference patterns showing the execution style."
    )
