"""Rule builder for converting EvaluationExample records into deterministic checks (Step 20).

Translates versioned evaluation benchmark records into machine-checkable rule suites
covering structure, requirement coverage, constraint preservation, output format,
and negative boundary compliance.
"""

import re
from typing import List, Optional, Set

from app.evaluation.check_schemas import (
    CheckType,
    CrispeSectionTarget,
    DeterministicCheckDefinition,
    EvaluationDimension,
)
from app.evaluation.schemas import EvaluationExample

# Stop words to filter out when extracting salient terms from natural language criteria
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "in", "to", "for",
    "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "should", "would", "must", "may", "might", "shall", "will", "that", "this",
    "these", "those", "it", "its", "they", "them", "their", "we", "our", "you",
    "your", "all", "any", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "just", "don", "shouldn", "prompt", "ai", "model", "must",
    "ensure", "include", "provide", "specify", "require", "request",
}

# Common format indicators mapped to search synonyms
FORMAT_INDICATOR_MAP = {
    "json": ["json", "javascript object notation", "json schema"],
    "markdown": ["markdown", "md", "##", "headers"],
    "table": ["table", "columns", "rows", "|"],
    "yaml": ["yaml", "yml"],
    "csv": ["csv", "comma-separated"],
    "bullet": ["bullet", "bulleted", "list", "bullets"],
    "step": ["step-by-step", "steps", "numbered list"],
    "code": ["code block", "snippet", "implementation"],
}


def _extract_salient_terms(text: str, max_terms: int = 5) -> List[str]:
    """Extract distinct salient words/phrases from natural language instruction text."""
    # Find word tokens of at least 3 letters
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]

    # Preserve uniqueness while maintaining order
    seen: Set[str] = set()
    salient: List[str] = []
    for token in filtered:
        if token not in seen:
            seen.add(token)
            salient.append(token)
        if len(salient) >= max_terms:
            break

    return salient


def _extract_format_indicators(format_str: str) -> List[str]:
    """Extract target indicators from expected_output_format description."""
    indicators: List[str] = []
    lower = format_str.lower()
    for key, syns in FORMAT_INDICATOR_MAP.items():
        if key in lower:
            indicators.extend(syns)

    if not indicators:
        # Fallback to salient terms from the format string
        indicators = _extract_salient_terms(format_str, max_terms=4)

    return list(set(indicators)) if indicators else ["format", "output"]


def build_checks_for_example(
    example: EvaluationExample,
    include_structural: bool = True,
) -> List[DeterministicCheckDefinition]:
    """Construct a comprehensive suite of deterministic checks for an evaluation example.

    If the example already defines explicit `deterministic_checks`, those are returned.
    Otherwise, standard checks are derived from the example's structured fields:
    1. CRISPE structural validity (all 6 sections present and non-empty).
    2. Section length requirements.
    3. Output format requirement.
    4. Critical requirements (marked critical with score capping).
    5. Constraints preservation.
    6. Prohibited behaviors (negative boundaries).

    Parameters:
        example: The benchmark evaluation record.
        include_structural: Whether to include canonical CRISPE structure checks.

    Returns:
        List of DeterministicCheckDefinition objects.
    """
    # If the example already contains explicit, author-defined deterministic checks, use them
    explicit_checks = getattr(example, "deterministic_checks", None)
    if explicit_checks:
        return list(explicit_checks)

    checks: List[DeterministicCheckDefinition] = []
    prefix = example.example_id

    # -------------------------------------------------------------------------
    # 1. Structural Checks (CRISPE Schema and Section Completeness)
    # -------------------------------------------------------------------------
    if include_structural:
        checks.append(
            DeterministicCheckDefinition(
                check_id=f"{prefix}_struct_valid",
                check_type=CheckType.CRISPE_SCHEMA_VALID,
                dimension=EvaluationDimension.STRUCTURE,
                target_section=CrispeSectionTarget.ALL,
                parameters={},
                required=True,
                is_critical=True,
                weight=2.0,
                description="Validates that the generated prompt satisfies canonical 6-section CRISPE schema.",
            )
        )

        for sec in (
            CrispeSectionTarget.CONTEXT,
            CrispeSectionTarget.ROLE,
            CrispeSectionTarget.INSTRUCTION,
            CrispeSectionTarget.SPECIFICATION,
            CrispeSectionTarget.PERFORMANCE,
            CrispeSectionTarget.EXAMPLE,
        ):
            checks.append(
                DeterministicCheckDefinition(
                    check_id=f"{prefix}_sec_{sec.value}",
                    check_type=CheckType.REQUIRED_SECTION_PRESENT,
                    dimension=EvaluationDimension.STRUCTURE,
                    target_section=sec,
                    parameters={"min_chars": 10},
                    required=True,
                    is_critical=False,
                    weight=0.5,
                    description=f"Verifies CRISPE section '{sec.value}' is populated with meaningful content.",
                )
            )

        # Minimum text length on instruction and specification
        checks.append(
            DeterministicCheckDefinition(
                check_id=f"{prefix}_len_instruction",
                check_type=CheckType.MINIMUM_TEXT_LENGTH,
                dimension=EvaluationDimension.STRUCTURE,
                target_section=CrispeSectionTarget.INSTRUCTION,
                parameters={"min_chars": 30},
                required=False,
                is_critical=False,
                weight=0.5,
                description="Verifies the instruction section has at least 30 characters.",
            )
        )
        checks.append(
            DeterministicCheckDefinition(
                check_id=f"{prefix}_len_specification",
                check_type=CheckType.MINIMUM_TEXT_LENGTH,
                dimension=EvaluationDimension.STRUCTURE,
                target_section=CrispeSectionTarget.SPECIFICATION,
                parameters={"min_chars": 20},
                required=False,
                is_critical=False,
                weight=0.5,
                description="Verifies the specification section has at least 20 characters.",
            )
        )

    # -------------------------------------------------------------------------
    # 2. Output Format Requirement
    # -------------------------------------------------------------------------
    if example.expected_output_format:
        indicators = _extract_format_indicators(example.expected_output_format)
        checks.append(
            DeterministicCheckDefinition(
                check_id=f"{prefix}_output_format",
                check_type=CheckType.OUTPUT_FORMAT_REQUIREMENT,
                dimension=EvaluationDimension.OUTPUT_FORMAT,
                target_section=CrispeSectionTarget.ALL,
                parameters={"format_indicators": indicators},
                required=True,
                is_critical=False,
                weight=1.0,
                description=f"Verifies output format requirement matches indicators: {indicators[:3]}",
            )
        )

    # -------------------------------------------------------------------------
    # 3. Critical Requirements (is_critical=True)
    # -------------------------------------------------------------------------
    for idx, crit in enumerate(example.critical_requirements):
        terms = _extract_salient_terms(crit, max_terms=4)
        if terms:
            checks.append(
                DeterministicCheckDefinition(
                    check_id=f"{prefix}_crit_{idx}",
                    check_type=CheckType.REQUIRED_TERM_GROUP,
                    dimension=EvaluationDimension.REQUIREMENTS,
                    target_section=CrispeSectionTarget.ALL,
                    parameters={"terms": terms, "min_matches": 1},
                    required=True,
                    is_critical=True,
                    weight=1.5,
                    description=f"Critical requirement: {crit[:80]}",
                )
            )

    # -------------------------------------------------------------------------
    # 4. Standard Acceptance Criteria / Behaviors
    # -------------------------------------------------------------------------
    for idx, req in enumerate(example.required_behaviors):
        terms = _extract_salient_terms(req, max_terms=3)
        if terms:
            checks.append(
                DeterministicCheckDefinition(
                    check_id=f"{prefix}_req_{idx}",
                    check_type=CheckType.REQUIRED_TERM_GROUP,
                    dimension=EvaluationDimension.REQUIREMENTS,
                    target_section=CrispeSectionTarget.ALL,
                    parameters={"terms": terms, "min_matches": 1},
                    required=True,
                    is_critical=False,
                    weight=1.0,
                    description=f"Required behavior: {req[:80]}",
                )
            )

    # -------------------------------------------------------------------------
    # 5. Constraints Preservation
    # -------------------------------------------------------------------------
    for idx, con in enumerate(example.constraints):
        terms = _extract_salient_terms(con, max_terms=3)
        if terms:
            checks.append(
                DeterministicCheckDefinition(
                    check_id=f"{prefix}_con_{idx}",
                    check_type=CheckType.CONSTRAINT_PRESERVED,
                    dimension=EvaluationDimension.CONSTRAINTS,
                    target_section=CrispeSectionTarget.ALL,
                    parameters={"terms": terms},
                    required=True,
                    is_critical=False,
                    weight=1.0,
                    description=f"Constraint preservation: {con[:80]}",
                )
            )

    # -------------------------------------------------------------------------
    # 6. Prohibited Behaviors (Negative Boundaries)
    # -------------------------------------------------------------------------
    for idx, pro in enumerate(example.prohibited_behaviors):
        terms = _extract_salient_terms(pro, max_terms=3)
        if terms:
            checks.append(
                DeterministicCheckDefinition(
                    check_id=f"{prefix}_pro_{idx}",
                    check_type=CheckType.FORBIDDEN_TEXT,
                    dimension=EvaluationDimension.PROHIBITIONS,
                    target_section=CrispeSectionTarget.ALL,
                    parameters={"forbidden_terms": terms, "allow_contextual_negation": True},
                    required=True,
                    is_critical=False,
                    weight=1.0,
                    description=f"Prohibited behavior avoidance: {pro[:80]}",
                )
            )

    return checks
