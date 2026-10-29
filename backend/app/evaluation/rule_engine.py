"""Deterministic rule execution engine for CRISPE prompt evaluation (Step 20).

Executes pure, machine-checkable assertion rules against generated CRISPE prompt
sections and rendered outputs, returning structured verdicts and diagnostic details.
"""

from enum import Enum
import re
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set, Tuple

from app.evaluation.check_schemas import (
    CheckOutcome,
    CheckResult,
    CheckType,
    CrispeSectionTarget,
    DeterministicCheckDefinition,
    EvaluationDimension,
)
from app.schemas.crispe import CrispeSections

# 6 canonical sections defined in CRISPE framework
CANONICAL_CRISPE_SECTIONS = (
    "context",
    "role",
    "instruction",
    "specification",
    "performance",
    "example",
)

# Known ReDoS danger patterns (nested quantifiers or overlapping alternation with repetition)
REDOS_SUSPECT_PATTERNS = (
    re.compile(r"\([^\)]*[\+\*]\)[\+\*]"),          # (a+)+ or (a*)*
    re.compile(r"\([^\)]*\|[^\)]*[\+\*]\)[\+\*]"),  # (a|b+)+
)

# Common negation prefixes that indicate the term is mentioned to prohibit it rather than endorse it
NEGATION_PREFIXES = (
    "do not",
    "don't",
    "never",
    "avoid",
    "must not",
    "should not",
    "cannot",
    "prohibit",
    "prohibited",
    "forbidden",
    "disallow",
    "no ",
    "without ",
)


def validate_safe_regex(pattern: str) -> Pattern[str]:
    """Validate and compile a regular expression, rejecting potential ReDoS patterns.

    Parameters:
        pattern: The regex pattern string.

    Returns:
        Compiled regex Pattern.

    Raises:
        ValueError: If pattern is malformed, too long, or structurally suspicious.
    """
    if not pattern or not isinstance(pattern, str):
        raise ValueError("Pattern must be a non-empty string.")

    if len(pattern) > 500:
        raise ValueError(f"Regex pattern exceeds maximum allowed length of 500 chars (len={len(pattern)}).")

    for danger in REDOS_SUSPECT_PATTERNS:
        if danger.search(pattern):
            raise ValueError(f"Potential catastrophic backtracking pattern detected in regex: '{pattern}'")

    try:
        return re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as exc:
        raise ValueError(f"Malformed regular expression '{pattern}': {exc}")


def validate_crispe_structure(sections: Any) -> Tuple[bool, List[str]]:
    """Verify whether the object satisfies the canonical 6-section CRISPE schema.

    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
    """
    issues: List[str] = []

    if sections is None:
        return False, ["Sections object is None."]

    if isinstance(sections, dict):
        for field_name in CANONICAL_CRISPE_SECTIONS:
            if field_name not in sections:
                issues.append(f"Missing required section: '{field_name}'.")
                continue
            val = sections[field_name]
            if not isinstance(val, str):
                issues.append(f"Section '{field_name}' must be a string, got {type(val).__name__}.")
            elif not val.strip():
                issues.append(f"Section '{field_name}' is empty or whitespace-only.")
        return len(issues) == 0, issues

    if isinstance(sections, CrispeSections):
        for field_name in CANONICAL_CRISPE_SECTIONS:
            val = getattr(sections, field_name, None)
            if not isinstance(val, str):
                issues.append(f"Section '{field_name}' must be a string, got {type(val).__name__}.")
            elif not val.strip():
                issues.append(f"Section '{field_name}' is empty or whitespace-only.")
        return len(issues) == 0, issues

    return False, [f"Object of type '{type(sections).__name__}' does not conform to CrispeSections."]


def extract_target_text(
    target: CrispeSectionTarget,
    sections: Optional[CrispeSections],
    raw_text: Optional[str] = None,
) -> str:
    """Extract the text corresponding to the requested CRISPE section."""
    if sections is None:
        return (raw_text or "").strip()

    if target == CrispeSectionTarget.CONTEXT:
        return (sections.context or "").strip()
    elif target == CrispeSectionTarget.ROLE:
        return (sections.role or "").strip()
    elif target == CrispeSectionTarget.INSTRUCTION:
        return (sections.instruction or "").strip()
    elif target == CrispeSectionTarget.SPECIFICATION:
        return (sections.specification or "").strip()
    elif target == CrispeSectionTarget.PERFORMANCE:
        return (sections.performance or "").strip()
    elif target == CrispeSectionTarget.EXAMPLE:
        return (sections.example or "").strip()
    elif target == CrispeSectionTarget.ALL:
        # Concatenate all 6 sections or use formatted markdown
        return (
            f"Context:\n{sections.context}\n\n"
            f"Role:\n{sections.role}\n\n"
            f"Instruction:\n{sections.instruction}\n\n"
            f"Specification:\n{sections.specification}\n\n"
            f"Performance:\n{sections.performance}\n\n"
            f"Example:\n{sections.example}"
        )
    return (raw_text or "").strip()


def normalize_text(text: str, case_sensitive: bool = False) -> str:
    """Normalize whitespace and optionally case for exact substring matching."""
    if not text:
        return ""
    # Collapse multiple whitespaces and tabs to single spaces
    collapsed = " ".join(text.split())
    return collapsed if case_sensitive else collapsed.lower()


def is_negated_mention(text: str, term: str) -> bool:
    """Determine whether term appears exclusively in a negative/prohibition context.

    For example, 'Do not use eval()' is a constraint instruction to avoid eval,
    not a violation directing the model to use eval.
    """
    lower_text = text.lower()
    lower_term = term.lower()

    pos = 0
    while True:
        idx = lower_text.find(lower_term, pos)
        if idx == -1:
            break

        # Check the 60 characters preceding the term
        start = max(0, idx - 60)
        prefix = lower_text[start:idx]

        has_negation = any(neg in prefix for neg in NEGATION_PREFIXES)
        if not has_negation:
            # Found at least one occurrence that was NOT preceded by a negation prefix
            return False

        pos = idx + len(lower_term)

    return True


def execute_check(
    check_def: DeterministicCheckDefinition,
    sections: Optional[CrispeSections],
    raw_text: Optional[str] = None,
) -> CheckResult:
    """Execute a single deterministic check against prompt sections.

    Pure, safe execution: catches unexpected exceptions and returns a CheckResult
    with outcome=ERROR rather than crashing.
    """
    check_id = check_def.check_id
    check_type = check_def.check_type
    dimension = check_def.dimension
    target = check_def.target_section
    params = check_def.parameters or {}
    weight = check_def.weight
    required = check_def.required
    is_critical = check_def.is_critical

    try:
        # 1. CRISPE_SCHEMA_VALID
        if check_type == CheckType.CRISPE_SCHEMA_VALID:
            is_valid, issues = validate_crispe_structure(sections)
            if is_valid:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition="All six CRISPE sections present, non-empty, and valid strings.",
                    actual_condition="Valid 6-section CRISPE prompt conforming to schema.",
                    diagnostic_message="Passed: Generated prompt satisfies canonical CRISPE schema.",
                )
            else:
                issue_summary = "; ".join(issues)
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition="All six CRISPE sections present, non-empty, and valid strings.",
                    actual_condition=f"Malformed CRISPE: {issue_summary}",
                    diagnostic_message=f"Failed: Canonical CRISPE schema validation failed ({issue_summary}).",
                )

        # 2. REQUIRED_SECTION_PRESENT
        if check_type == CheckType.REQUIRED_SECTION_PRESENT:
            target_str = extract_target_text(target, sections, raw_text)
            min_chars = params.get("min_chars", 5)
            section_name = target.value if isinstance(target, CrispeSectionTarget) else str(target)
            if len(target_str.strip()) >= min_chars:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Section '{section_name}' present with >= {min_chars} characters.",
                    actual_condition=f"Section '{section_name}' has {len(target_str.strip())} characters.",
                    diagnostic_message=f"Passed: Section '{section_name}' is meaningfully populated.",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Section '{section_name}' present with >= {min_chars} characters.",
                    actual_condition=f"Section '{section_name}' has {len(target_str.strip())} characters.",
                    diagnostic_message=f"Failed: Section '{section_name}' is empty or shorter than {min_chars} chars.",
                )

        # 3. MINIMUM_TEXT_LENGTH
        if check_type == CheckType.MINIMUM_TEXT_LENGTH:
            target_str = extract_target_text(target, sections, raw_text)
            min_chars = params.get("min_chars")
            min_words = params.get("min_words")

            actual_chars = len(target_str.strip())
            actual_words = len(target_str.strip().split())

            passed = True
            expected_parts: List[str] = []
            actual_parts: List[str] = []

            if min_chars is not None:
                expected_parts.append(f">= {min_chars} chars")
                actual_parts.append(f"{actual_chars} chars")
                if actual_chars < min_chars:
                    passed = False

            if min_words is not None:
                expected_parts.append(f">= {min_words} words")
                actual_parts.append(f"{actual_words} words")
                if actual_words < min_words:
                    passed = False

            expected_desc = " and ".join(expected_parts)
            actual_desc = ", ".join(actual_parts)

            if passed:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Target text length {expected_desc}.",
                    actual_condition=f"Target text length {actual_desc}.",
                    diagnostic_message="Passed: Text satisfies minimum length requirements.",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Target text length {expected_desc}.",
                    actual_condition=f"Target text length {actual_desc}.",
                    diagnostic_message=f"Failed: Text does not meet length requirement ({actual_desc} vs expected {expected_desc}).",
                )

        # 4. REQUIRED_TEXT
        if check_type == CheckType.REQUIRED_TEXT:
            required_phrase = params.get("text", "")
            case_sensitive = params.get("case_sensitive", False)
            target_str = extract_target_text(target, sections, raw_text)

            norm_target = normalize_text(target_str, case_sensitive=case_sensitive)
            norm_phrase = normalize_text(required_phrase, case_sensitive=case_sensitive)

            if norm_phrase in norm_target:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Presence of phrase '{required_phrase}'.",
                    actual_condition=f"Phrase '{required_phrase}' detected in target section.",
                    diagnostic_message=f"Passed: Explicit phrase '{required_phrase}' was found.",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Presence of phrase '{required_phrase}'.",
                    actual_condition=f"Phrase '{required_phrase}' was not found in target section.",
                    diagnostic_message=f"Failed: Required phrase '{required_phrase}' is missing.",
                )

        # 5. REQUIRED_TERM_GROUP
        if check_type == CheckType.REQUIRED_TERM_GROUP:
            terms: List[str] = params.get("terms", [])
            min_matches = params.get("min_matches", 1)
            case_sensitive = params.get("case_sensitive", False)
            target_str = extract_target_text(target, sections, raw_text)

            norm_target = normalize_text(target_str, case_sensitive=case_sensitive)

            matched_terms: List[str] = []
            for t in terms:
                norm_term = normalize_text(t, case_sensitive=case_sensitive)
                if norm_term in norm_target:
                    matched_terms.append(t)

            if len(matched_terms) >= min_matches:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"At least {min_matches} of terms: {terms}.",
                    actual_condition=f"Matched {len(matched_terms)} term(s): {matched_terms}.",
                    diagnostic_message=f"Passed: Term group requirement satisfied with matches {matched_terms}.",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"At least {min_matches} of terms: {terms}.",
                    actual_condition=f"Matched only {len(matched_terms)} term(s): {matched_terms}.",
                    diagnostic_message=f"Failed: Insufficient matches from term group {terms}.",
                )

        # 6. FORBIDDEN_TEXT
        if check_type == CheckType.FORBIDDEN_TEXT:
            forbidden_terms: List[str] = (
                params.get("forbidden_terms")
                or params.get("terms")
                or ([params.get("text")] if params.get("text") else [])
            )
            allow_negation = params.get("allow_contextual_negation", True)
            target_str = extract_target_text(target, sections, raw_text)
            norm_target = target_str.lower()

            violated_terms: List[str] = []
            for ft in forbidden_terms:
                lower_term = ft.lower()
                if lower_term in norm_target:
                    if allow_negation and is_negated_mention(target_str, ft):
                        # Mentioned only in negative/prohibitive context (e.g. 'do not use X')
                        continue
                    violated_terms.append(ft)

            if not violated_terms:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Absence of prohibited patterns: {forbidden_terms}.",
                    actual_condition="No prohibited patterns detected.",
                    diagnostic_message="Passed: No prohibited behaviors or phrases detected.",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Absence of prohibited patterns: {forbidden_terms}.",
                    actual_condition=f"Prohibited pattern(s) detected: {violated_terms}.",
                    diagnostic_message=f"Failed: Prompt contains prohibited term(s) {violated_terms}.",
                )

        # 7. REGEX_CONSTRAINT
        if check_type == CheckType.REGEX_CONSTRAINT:
            pattern_str = params.get("pattern", "")
            compiled_re = validate_safe_regex(pattern_str)
            target_str = extract_target_text(target, sections, raw_text)

            match = compiled_re.search(target_str)
            if match:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Match regular expression: {pattern_str}.",
                    actual_condition=f"Pattern matched at index {match.start()}.",
                    diagnostic_message=f"Passed: Target text matches pattern '{pattern_str}'.",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Match regular expression: {pattern_str}.",
                    actual_condition="No pattern match found.",
                    diagnostic_message=f"Failed: Target text does not match pattern '{pattern_str}'.",
                )

        # 8. CONSTRAINT_PRESERVED
        if check_type == CheckType.CONSTRAINT_PRESERVED:
            target_str = extract_target_text(target, sections, raw_text)
            terms: Optional[List[str]] = params.get("terms")
            pattern_str = params.get("pattern")
            text_phrase = params.get("text")

            preserved = False
            actual_info = ""

            if terms:
                norm_target = target_str.lower()
                matches = [t for t in terms if t.lower() in norm_target]
                preserved = len(matches) > 0
                actual_info = f"Matched constraint terms: {matches}" if preserved else "No constraint terms matched."
            elif pattern_str:
                compiled = validate_safe_regex(pattern_str)
                preserved = bool(compiled.search(target_str))
                actual_info = f"Pattern '{pattern_str}' matched." if preserved else "Pattern did not match."
            elif text_phrase:
                preserved = text_phrase.lower() in target_str.lower()
                actual_info = f"Phrase '{text_phrase}' found." if preserved else f"Phrase '{text_phrase}' missing."
            else:
                # If no machine-checkable terms or pattern, check length/non-empty
                preserved = len(target_str.strip()) > 0
                actual_info = "Content present."

            if preserved:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Preservation of constraint: {check_def.description}",
                    actual_condition=actual_info,
                    diagnostic_message=f"Passed: {check_def.description}",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Preservation of constraint: {check_def.description}",
                    actual_condition=actual_info,
                    diagnostic_message=f"Failed: Constraint was not preserved ({actual_info}).",
                )

        # 9. OUTPUT_FORMAT_REQUIREMENT
        if check_type == CheckType.OUTPUT_FORMAT_REQUIREMENT:
            indicators: List[str] = params.get("format_indicators") or [params.get("format_name", "")]
            target_str = extract_target_text(target, sections, raw_text)
            norm_target = target_str.lower()

            matches = [ind for ind in indicators if ind and ind.lower() in norm_target]
            if matches:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.PASSED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Specification of output format matching: {indicators}.",
                    actual_condition=f"Matched format indicators: {matches}.",
                    diagnostic_message=f"Passed: Output format requirement declared ({matches}).",
                )
            else:
                return CheckResult(
                    check_id=check_id,
                    check_type=check_type,
                    dimension=dimension,
                    outcome=CheckOutcome.FAILED,
                    required=required,
                    is_critical=is_critical,
                    weight=weight,
                    expected_condition=f"Specification of output format matching: {indicators}.",
                    actual_condition="No format indicators detected.",
                    diagnostic_message=f"Failed: Output format requirement missing (expected one of {indicators}).",
                )

        # Unsupported check type
        return CheckResult(
            check_id=check_id,
            check_type=check_type,
            dimension=dimension,
            outcome=CheckOutcome.ERROR,
            required=required,
            is_critical=is_critical,
            weight=weight,
            expected_condition="Supported check type execution.",
            actual_condition=f"Unknown check type '{check_type}'.",
            diagnostic_message=f"Execution error: Unsupported check type '{check_type}'.",
            execution_error=f"Unsupported check type: {check_type}",
        )

    except Exception as exc:
        return CheckResult(
            check_id=check_id,
            check_type=check_type,
            dimension=dimension,
            outcome=CheckOutcome.ERROR,
            required=required,
            is_critical=is_critical,
            weight=weight,
            expected_condition="Successful check execution.",
            actual_condition=f"Exception raised: {type(exc).__name__}: {str(exc)}",
            diagnostic_message=f"Execution error during check '{check_id}': {str(exc)}",
            execution_error=f"{type(exc).__name__}: {str(exc)}",
        )
