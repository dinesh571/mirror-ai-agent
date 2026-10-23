"""Conservative response refinement using supported offline templates."""

import re

from models import ReflectionReport, RefinementResult, RequestKind
from reflection_engine import ReflectionEngine
from response_generator import (
    ResponseGenerator,
    classify_request,
    detect_programming_language,
    response_matches_requested_language,
)


class ResponseRefiner:
    def __init__(
        self,
        generator: ResponseGenerator,
        reflection_engine: ReflectionEngine,
    ) -> None:
        self._generator = generator
        self._reflection_engine = reflection_engine

    def refine(
        self, request: str, response: str, report: ReflectionReport
    ) -> RefinementResult:
        candidate = self._generator.generate_deterministic(request)
        candidate_report = (
            self._reflection_engine.evaluate(request, candidate)
            if candidate is not None
            else None
        )
        if (
            candidate is not None
            and candidate.strip()
            and candidate.strip() != response.strip()
            and candidate_report is not None
            and candidate_report.score > report.score
            and (
                detect_programming_language(request) is None
                or response_matches_requested_language(request, candidate) is True
            )
        ):
            return RefinementResult(
                response=candidate,
                changed=True,
                explanation=(
                    "Replaced the incomplete response with a supported offline "
                    "example that scores better under the same rule-based checks."
                ),
            )

        if not response.strip():
            if (
                candidate is not None
                and candidate.strip()
                and candidate_report is not None
                and candidate_report.score > report.score
                and (
                    detect_programming_language(request) is None
                    or response_matches_requested_language(request, candidate) is True
                )
            ):
                return RefinementResult(
                    response=candidate,
                    changed=True,
                    explanation="Generated a supported offline response for the empty answer.",
                )
            return RefinementResult(
                response="No answer could be generated for this request.",
                changed=True,
                explanation="Replaced the empty response with a clear status message.",
            )

        if not report.suggestions:
            return RefinementResult(
                response=response,
                changed=False,
                explanation=(
                    "No meaningful improvement was identified using the available "
                    "offline rules; the original response is preserved."
                ),
            )

        if classify_request(request) is RequestKind.PYTHON:
            formatted = self._format_unfenced_code(response)
            formatted_report = self._reflection_engine.evaluate(request, formatted)
            if formatted != response and formatted_report.score > report.score:
                return RefinementResult(
                    response=formatted,
                    changed=True,
                    explanation="Formatted the existing code for readability.",
                )

        reason = (
            "No candidate improved the supported evaluation checks, so the original "
            "response is preserved."
            if candidate is None
            or candidate.strip() == response.strip()
            or candidate_report is None
            or candidate_report.score <= report.score
            else
            "The available offline provider cannot safely address the remaining suggestions, so the original response is preserved."
        )
        return RefinementResult(
            response=response,
            changed=False,
            explanation=reason,
        )

    @staticmethod
    def _format_unfenced_code(response: str) -> str:
        if "```" in response:
            return response
        lines = response.strip().splitlines()
        if len(lines) != 1 or not re.search(
            r"^\s*(?:def\s+\w+|print\s*\(|(?:from|import)\s+)", lines[0]
        ):
            return response
        return (
            "Here is the existing code, formatted as a Python example:\n\n"
            f"```python\n{lines[0].strip()}\n```"
        )
