"""Intent-aware deterministic checks for MIRROR responses."""

import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

from models import (
    ConversationIntent,
    CybersecurityIntent,
    EvaluationFinding,
    KnowledgeIntent,
    ReflectionReport,
    RequestKind,
    ResponseStatus,
)
from response_generator import (
    calculate_arithmetic,
    classify_request,
    detect_conversation_intent,
    detect_cybersecurity_intent,
    detect_knowledge_intent,
    detect_programming_language,
    is_division_by_zero,
    is_limitation_response,
    response_has_requested_language_fence,
    response_matches_requested_language,
)


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "do", "for", "how",
    "i", "in", "is", "it", "me", "of", "on", "or", "please", "the",
    "to", "what", "with", "write", "you", "python", "program", "code",
    "function", "script", "number", "numbers", "explain", "why", "need",
}
_CODE_PATTERN = re.compile(
    r"^\s*(?:"
    r"def\s+\w+\s*\(|class\s+\w+|#include\b|"
    r"(?:if|elif|else|for|while|return|import|from|print|printf|"
    r"console\.log|SELECT)\b|"
    r"int\s+main\s*\(|public\s+static\s+void\s+main|"
    r"[A-Za-z_]\w*\s*(?::[^=]+)?="
    r")",
    re.M,
)
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


class ReflectionEngine:
    """Evaluate only properties that simple offline rules can inspect."""

    def evaluate(self, request: str, response: str) -> ReflectionReport:
        kind = classify_request(request)
        conversation = detect_conversation_intent(request)
        knowledge = detect_knowledge_intent(request)
        cybersecurity = detect_cybersecurity_intent(request)
        text = response.strip()
        empty = not text
        fallback = is_limitation_response(text)
        findings: List[EvaluationFinding] = []

        findings.append(self._finding(
            "non_empty", 5, 5 if text else 0,
            "The response contains text." if text else
            "No useful answer was provided because the response is empty.",
        ))

        if conversation is not None:
            relevant = self._conversation_reply_matches(conversation, text)
            task_evidence = relevant
            partial_evidence = False
            findings.append(self._finding(
                "addresses_request", 25, 22 if relevant else 0,
                "The reply matches the conversational intent; phrase matching cannot assess every nuance."
                if relevant else
                "The response does not appropriately address this conversational input.",
            ))
            findings.append(self._finding(
                "task_completion", 25, 22 if relevant else 0,
                "The conversational input receives an appropriate reply."
                if relevant else "The conversational input remains unanswered.",
            ))
            findings.append(self._finding(
                "conversational_appropriateness", 15, 13 if relevant else 0,
                "The response uses a recognizable, natural conversational phrase."
                if relevant else "No suitable conversational response was identified.",
            ))
            code_requested = False
        elif kind is RequestKind.ARITHMETIC:
            expected = calculate_arithmetic(request)
            actual = self._arithmetic_answer(text)
            zero_division = is_division_by_zero(request)
            correct_zero_handling = zero_division and (
                "division by zero" in text.casefold()
                and "undefined" in text.casefold()
            )
            try:
                correct = (
                    correct_zero_handling
                    or expected is not None
                    and actual == Decimal(expected[1])
                )
            except InvalidOperation:
                correct = False
            attempted = self._contains_calculation_attempt(text)
            relevant = correct or (
                expected is not None
                and all(operand in text for operand in expected[0].split()[::2])
            )
            task_evidence = correct
            partial_evidence = relevant and not correct
            findings.append(self._finding(
                "addresses_request", 25,
                25 if relevant and correct or correct_zero_handling
                else 17 if relevant else 0,
                "The response addresses the requested calculation."
                if relevant or correct_zero_handling else "No relevant calculation result was identified.",
            ))
            findings.append(self._finding(
                "task_completion", 25, 25 if correct else 4 if attempted else 0,
                "The arithmetic result matches the deterministically calculated result."
                if correct and not zero_division else
                "The response correctly explains that division by zero is undefined."
                if correct_zero_handling else
                "The calculation is attempted, but its result is incorrect."
                if attempted else "The requested calculation was not completed.",
            ))
            findings.append(self._finding(
                "factual_correctness", 25, 25 if correct else 4 if attempted else 0,
                "The numeric answer matches the supported arithmetic calculation."
                if correct and not zero_division else
                "The response correctly identifies division by zero as undefined."
                if correct_zero_handling else
                "The numeric answer does not match the supported arithmetic calculation."
                if attempted else "No answer was available for a correctness check.",
            ))
            code_requested = False
        elif kind is RequestKind.PYTHON and knowledge is None and cybersecurity is None:
            has_code = self._has_plausible_code(text)
            language_match = response_matches_requested_language(request, text)
            request_terms = self._keywords(request)
            response_terms = self._code_keywords(text)
            hello_world_requested = (
                "hello" in request.casefold() and "world" in request.casefold()
            )
            hello_world_present = bool(
                re.search(
                    r"(?:print|printf|console\.log|System\.out\.print|std::cout)"
                    r"[^;\n]*"
                    r"hello\s*,?\s*world",
                    text,
                    re.I,
                )
            )
            output_match = not hello_world_requested or hello_world_present
            relevant = (
                has_code
                and bool(request_terms & response_terms)
                and language_match is not False
                and output_match
            )
            task_evidence = relevant
            partial_evidence = False
            findings.append(self._finding(
                "addresses_request", 25, 23 if relevant else 10 if has_code else 0,
                "The code example includes terms relevant to the programming request."
                if relevant else
                "The response does not implement the requested Hello World output."
                if has_code and not output_match else
                "The response code does not match the explicitly requested language."
                if language_match is False else
                "Code is present, but its relevance to the requested solution is unclear."
                if has_code else "No code example addresses the programming request.",
            ))
            findings.append(self._finding(
                "task_completion", 25, 22 if relevant else 8 if has_code else 0,
                "A relevant code example appears to complete the programming request."
                if relevant else
                "The requested language or output is missing from the code."
                if language_match is False or not output_match else
                "A code-shaped response is present, but task completion is unclear."
                if has_code else "The programming request remains incomplete without code.",
            ))
            requested_language = detect_programming_language(request)
            if requested_language is not None:
                findings.append(self._finding(
                    "requested_language",
                    35,
                    35 if language_match else 0,
                    "The code matches the explicitly requested programming language."
                    if language_match else
                    "The code does not match the explicitly requested programming language.",
                ))
                correctly_fenced = (
                    response_has_requested_language_fence(request, text) is True
                )
                findings.append(self._finding(
                    "code_format",
                    10,
                    10 if has_code and correctly_fenced else 0,
                    "The code is fenced with the requested language label."
                    if has_code and correctly_fenced else
                    "The code is not fenced with the requested language label.",
                ))
            findings.append(self._finding(
                "plausible_code", 20, 20 if has_code else 0,
                "A code-shaped example is present; this does not prove it is correct."
                if has_code else "No plausible code example was identified.",
            ))
            formatted_code = "```" in text
            findings.append(self._finding(
                "clarity", 15, 15 if has_code and formatted_code
                and language_match is not False else 11 if has_code else 0 if empty else 6,
                "The code is clearly formatted."
                if has_code and formatted_code and language_match is not False else
                "Code is present but could be more clearly formatted."
                if has_code else "The response does not clearly present a solution.",
            ))
            code_requested = True
        elif cybersecurity is not None:
            relevant, completed, cybersecurity_findings = self._evaluate_cybersecurity(
                cybersecurity, text, fallback
            )
            task_evidence = relevant and completed
            partial_evidence = relevant and not completed and not fallback
            findings.extend(cybersecurity_findings)
            code_requested = False
        elif knowledge is not None:
            expected_terms = self._knowledge_terms(knowledge)
            response_terms = self._keywords(text)
            matching_terms = expected_terms & response_terms
            relevant = bool(matching_terms) and not fallback
            task_evidence = relevant
            partial_evidence = False
            findings.append(self._finding(
                "addresses_request", 25,
                min(25, round(25 * len(matching_terms) / len(expected_terms)))
                if relevant else 0,
                "The response includes key concepts expected for this supported topic."
                if relevant else
                "The response does not include the expected concepts for this topic.",
            ))
            findings.append(self._finding(
                "task_completion", 25, 23 if relevant else 0,
                "The expected core concepts for this offline topic are present; this is a heuristic, not proof."
                if relevant else "The supported question remains unanswered.",
            ))
            findings.append(self._finding(
                "reference_alignment", 15, 12 if relevant else 0,
                "Key concepts align with the offline topic reference; this does not prove factual correctness."
                if relevant else "No supported reference concepts were found in the answer.",
            ))
            code_requested = False
        else:
            relevant, overlap = self._general_relevance(request, text)
            task_evidence = False
            partial_evidence = relevant and not fallback
            findings.append(self._finding(
                "addresses_request", 25, overlap,
                "Some request-specific terms appear in the response; this keyword check cannot establish semantic relevance."
                if overlap else
                "The response does not connect to the request under the keyword check.",
            ))
            findings.append(self._finding(
                "task_completion", 25, 0 if fallback else 12 if relevant else 0,
                "The request appears partially addressed by keyword checks, not verified for completeness."
                if relevant and not fallback else
                "The requested task remains unanswered by the limitation statement."
                if fallback else "The requested task remains unanswered.",
            ))
            code_requested = False

        if conversation is None:
            clarity_kind = (
                RequestKind.GENERAL
                if knowledge is not None or cybersecurity is not None
                else kind
            )
            clarity_points, clarity_text = self._clarity(
                text, clarity_kind, fallback
            )
            findings.append(self._finding(
                "clarity", 15, clarity_points, clarity_text,
            ))

        detail_needed = self._requires_detail(request) and cybersecurity is None
        if fallback:
            detail_needed = True
        if detail_needed:
            has_alternative = self._has_useful_alternative(text)
            has_detail = has_alternative if fallback else self._has_relevant_detail(text)
            findings.append(self._finding(
                "useful_detail", 10, 8 if has_detail else 0,
                "A relevant next step or useful detail is offered."
                if has_detail else
                "The response lacks details or a useful next step for this request.",
            ))

        if fallback:
            honest = self._explains_limitation(text)
            has_alternative = self._has_useful_alternative(text)
            findings.append(self._finding(
                "honesty", 10, 10 if honest else 0,
                "The response clearly acknowledges the offline provider's limitation."
                if honest else "The limitation is not stated clearly.",
            ))
            findings.append(self._finding(
                "useful_alternative", 10, 10 if has_alternative else 0,
                "A relevant alternative is offered."
                if has_alternative else "No useful alternative is offered.",
            ))

        if cybersecurity is not None:
            findings.append(EvaluationFinding(
                criterion="factual_correctness",
                passed=False,
                explanation=(
                    "These offline checks assess topic coverage, not the factual "
                    "correctness of cybersecurity advice."
                ),
                applicable=False,
            ))

        if code_requested:
            actionability = bool(_CODE_PATTERN.search(text))
            findings.append(self._finding(
                "actionability", 10, 10 if actionability else 0,
                "The code example provides an actionable starting point."
                if actionability else "No actionable programming steps or code were provided.",
            ))
        elif self._requires_actionability(request):
            actionable = self._is_actionable(text)
            findings.append(self._finding(
                "actionability", 10, 10 if actionable else 0,
                "The response includes concrete steps or a usable next action."
                if actionable else "The request asks for guidance, but concrete steps are missing.",
            ))

        if kind is RequestKind.PYTHON and knowledge is None and cybersecurity is None:
            findings.append(EvaluationFinding(
                criterion="factual_correctness",
                passed=False,
                explanation="Python correctness cannot be established by these code-shape rules.",
                applicable=False,
            ))
        elif (
            kind is not RequestKind.ARITHMETIC
            and knowledge is None
            and cybersecurity is None
        ):
            findings.append(EvaluationFinding(
                criterion="factual_correctness",
                passed=False,
                explanation="Factual correctness is not deterministically verifiable for this request type.",
                applicable=False,
            ))

        max_points = sum(item.weight for item in findings if item.applicable)
        awarded = sum(item.points_awarded for item in findings if item.applicable)
        score = round(awarded * 100 / max_points) if max_points else 0
        score = max(0, min(100, score))
        if (
            kind is RequestKind.PYTHON
            and detect_programming_language(request) is not None
            and response_matches_requested_language(request, text) is False
        ):
            score = min(score, 30)
        strengths = tuple(
            dict.fromkeys(item.explanation for item in findings
            if item.applicable
            and item.criterion != "non_empty"
            and item.passed
            )
        )
        weaknesses = tuple(
            dict.fromkeys(item.explanation for item in findings
            if item.applicable and not item.passed
            )
        )
        suggestions = self._suggestions(findings)
        if empty:
            status = ResponseStatus.EMPTY
        elif fallback:
            status = ResponseStatus.LIMITATION
        elif task_evidence:
            status = ResponseStatus.ANSWERED
        elif partial_evidence:
            status = ResponseStatus.PARTIAL
        else:
            status = ResponseStatus.IRRELEVANT

        return ReflectionReport(
            score=score,
            response_status=status,
            findings=tuple(findings),
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
        )

    @staticmethod
    def _finding(
        criterion: str, weight: int, points: int, explanation: str,
    ) -> EvaluationFinding:
        return EvaluationFinding(
            criterion=criterion,
            passed=points >= round(weight * 0.8),
            explanation=explanation,
            deduction=weight - points,
            applicable=True,
            weight=weight,
            points_awarded=points,
        )

    @staticmethod
    def _keywords(text: str) -> set[str]:
        return {
            word.casefold()
            for word in re.findall(r"[A-Za-z0-9]+", text)
            if word.casefold() not in _STOP_WORDS and len(word) > 1
        }

    @classmethod
    def _code_keywords(cls, text: str) -> set[str]:
        blocks = re.findall(r"```[A-Za-z+#]*\s*(.*?)```", text, re.S)
        code = "\n".join(blocks) if blocks else "\n".join(
            line for line in text.splitlines() if _CODE_PATTERN.match(line)
        )
        return cls._keywords(code)

    @staticmethod
    def _has_plausible_code(text: str) -> bool:
        blocks = re.findall(r"```[A-Za-z+#]*\s*(.*?)```", text, re.I | re.S)
        candidates = blocks if blocks else text.splitlines()
        return any(_CODE_PATTERN.search(candidate) for candidate in candidates)

    @staticmethod
    def _knowledge_terms(intent: KnowledgeIntent) -> set[str]:
        terms = {
            KnowledgeIntent.AGI: {
                "artificial", "general", "intelligence", "broad", "tasks",
                "narrow", "systems",
            },
            KnowledgeIntent.AI_VS_ML: {
                "artificial", "intelligence", "broader", "machine", "learning",
                "approach", "data", "part",
            },
            KnowledgeIntent.MACHINE_LEARNING: {
                "machine", "learning", "data", "patterns", "algorithms",
            },
            KnowledgeIntent.AI: {
                "artificial", "intelligence", "systems", "tasks",
            },
            KnowledgeIntent.PYTHON: {
                "python", "programming", "language", "readable", "syntax",
            },
            KnowledgeIntent.DATABASE: {
                "database", "data", "organized", "accessed", "managed", "updated",
            },
            KnowledgeIntent.OPERATING_SYSTEM: {
                "operating", "system", "hardware", "applications", "processes",
                "memory", "files",
            },
            KnowledgeIntent.PYTHON_LIST_TUPLE: {
                "list", "mutable", "tuple", "immutable", "ordered", "sequence",
            },
        }
        return terms[intent]

    @classmethod
    def _evaluate_cybersecurity(
        cls,
        intent: CybersecurityIntent,
        response: str,
        fallback: bool,
    ) -> Tuple[bool, bool, List[EvaluationFinding]]:
        lowered = response.casefold()
        expected = {
            CybersecurityIntent.INTRODUCTION: (
                ("protect", "protection"),
                ("computer", "phone", "network"),
                ("attack",),
                ("password",),
                ("two-factor", "two factor"),
                ("update",),
                ("suspicious", "phishing"),
                ("backup", "back up"),
                ("risk", "guarantee"),
            ),
            CybersecurityIntent.PROTECT_COMPUTER: (
                ("update",),
                ("password",),
                ("two-factor", "two factor"),
                ("security software",),
                ("suspicious", "link"),
                ("backup", "back up"),
            ),
            CybersecurityIntent.PHISHING: (
                ("impersonat", "pretend", "trusted"),
                ("trick",),
                ("information", "credential"),
                ("link", "attachment"),
                ("verify", "cautious", "avoid"),
            ),
            CybersecurityIntent.STRONG_PASSWORD: (
                ("long",),
                ("unique",),
                ("password",),
                ("reuse", "re-use", "each account"),
                ("manager", "two-factor", "two factor"),
            ),
            CybersecurityIntent.TWO_FACTOR: (
                ("second",),
                ("password",),
                ("code", "security key", "verification"),
                ("account",),
            ),
        }[intent]
        covered = sum(any(term in lowered for term in group) for group in expected)
        coverage = covered / len(expected)
        relevant = not fallback and coverage >= 0.4
        complete = relevant and coverage >= 0.65
        task_points = round(35 * coverage) if relevant else 0
        findings = [
            cls._finding(
                "addresses_request",
                25,
                round(25 * coverage) if relevant else 0,
                (
                    "The response covers concepts relevant to the requested cybersecurity topic."
                    if relevant
                    else "The response does not meaningfully address the cybersecurity question."
                ),
            ),
            cls._finding(
                "task_completion",
                35,
                task_points,
                (
                    "The response covers the core points expected for this question."
                    if complete
                    else "The response is missing relevant points for this cybersecurity question."
                ),
            ),
            cls._finding(
                "beginner_detail",
                15,
                15 if coverage >= 0.65 else round(15 * coverage),
                (
                    "The response provides relevant beginner-level explanation or advice."
                    if coverage >= 0.65
                    else "Some beginner-level explanation or advice relevant to the question is missing."
                ),
            ),
        ]
        actionable_intents = {
            CybersecurityIntent.INTRODUCTION,
            CybersecurityIntent.PROTECT_COMPUTER,
            CybersecurityIntent.STRONG_PASSWORD,
        }
        if intent in actionable_intents:
            actionable = (
                bool(re.search(r"(?:\b1[.)]|\bfirst\b)", response))
                or "use " in lowered
                or "enable " in lowered
                or "avoid " in lowered
            )
            findings.append(cls._finding(
                "actionability",
                10,
                10 if actionable else 0,
                "The response provides practical protective steps."
                if actionable
                else "The response lacks a practical next step for this safety question.",
            ))
        findings.append(EvaluationFinding(
            criterion="factual_correctness",
            passed=False,
            explanation=(
                "The offline cybersecurity checks measure expected topic coverage, "
                "not factual correctness."
            ),
            applicable=False,
        ))
        return relevant, complete, findings

    @classmethod
    def _general_relevance(cls, request: str, response: str) -> Tuple[bool, int]:
        requested = cls._keywords(request)
        if not response or not requested or is_limitation_response(response):
            return False, 0
        overlap = requested & cls._keywords(response)
        if not overlap:
            return False, 0
        fraction = len(overlap) / len(requested)
        return True, min(20, max(8, round(20 * fraction)))

    @staticmethod
    def _arithmetic_answer(response: str) -> Optional[Decimal]:
        candidate = response.split("=", 1)[1] if "=" in response else response
        matches = _NUMBER_PATTERN.findall(candidate.strip().rstrip(".!?"))
        if not matches:
            return None
        try:
            return Decimal(matches[-1])
        except InvalidOperation:
            return None

    @staticmethod
    def _contains_calculation_attempt(response: str) -> bool:
        return "=" in response and bool(_NUMBER_PATTERN.search(response))

    @staticmethod
    def _clarity(
        response: str, kind: RequestKind, fallback: bool
    ) -> Tuple[int, str]:
        if not response:
            return 0, "The response is empty and cannot be read."
        if kind is RequestKind.ARITHMETIC:
            return 15, "The calculation result is concise and readable."
        if kind is RequestKind.PYTHON:
            if "```" in response:
                return 15, "The response is organized with a formatted code block."
            return 11, "The code is present but could be more clearly formatted."
        if fallback and any(mark in response for mark in ".!?"):
            return 13, "The limitation and next step are stated in readable language."
        if len(response) < 12:
            return 8, "The concise response is readable, though its meaning is limited."
        if any(mark in response for mark in ".!?\n"):
            return 13, "The response uses readable sentence structure."
        return 8, "The response is present but would be clearer as a complete sentence."

    @staticmethod
    def _conversation_reply_matches(
        intent: ConversationIntent, response: str
    ) -> bool:
        patterns = {
            ConversationIntent.GREETING: r"\b(?:hello|hi|hey|good morning|good afternoon|good evening)\b",
            ConversationIntent.WELL_BEING: r"\b(?:ready to help|doing well|i'm fine|i am fine)\b",
            ConversationIntent.GRATITUDE: r"\b(?:you're welcome|you are welcome|no problem|happy to help)\b",
            ConversationIntent.FAREWELL: r"\b(?:goodbye|bye|take care|see you)\b",
        }
        return bool(re.search(patterns[intent], response, re.IGNORECASE))

    @staticmethod
    def _requires_detail(request: str) -> bool:
        lowered = request.casefold()
        return any(
            cue in lowered
            for cue in ("explain", "describe", "why", "how does", "steps", "instructions")
        )

    @staticmethod
    def _requires_actionability(request: str) -> bool:
        lowered = request.casefold()
        return any(
            cue in lowered
            for cue in ("how do i", "how to", "steps", "instructions", "guide")
        )

    @staticmethod
    def _has_relevant_detail(response: str) -> bool:
        return bool(re.search(r"(?:\b1[.)]|\bfirst\b|\bthen\b|\bfor example\b|```)", response, re.I))

    @staticmethod
    def _has_useful_alternative(response: str) -> bool:
        lowered = response.casefold()
        return any(
            cue in lowered
            for cue in (
                "if you share", "paste", "provide a source", "article",
                "summarize", "simple arithmetic", "basic python",
            )
        )

    @staticmethod
    def _explains_limitation(response: str) -> bool:
        lowered = response.casefold()
        return any(
            cue in lowered
            for cue in ("can't verify", "cannot verify", "limited knowledge",
                        "can't reliably answer", "cannot reliably answer")
        )

    @staticmethod
    def _is_actionable(response: str) -> bool:
        return bool(re.search(r"(?:\b1[.)]|\bfirst\b|\bthen\b|```)", response, re.I))

    @staticmethod
    def _suggestions(findings: List[EvaluationFinding]) -> Tuple[str, ...]:
        advice = {
            "non_empty": "Provide a response to the request.",
            "addresses_request": "Make the connection to the user's request explicit.",
            "task_completion": "Complete the requested task, or explain its limitation and offer a relevant next step.",
            "conversational_appropriateness": "Use a natural reply suited to the conversational intent.",
            "factual_correctness": "Check the answer against the supported arithmetic operation.",
            "reference_alignment": "Include the core concepts from the offline topic reference; this is not factual verification.",
            "beginner_detail": "Add the missing core explanation or safety details relevant to the question.",
            "clarity": "Improve readability while keeping the answer appropriately concise.",
            "useful_detail": "Add relevant detail or a useful next step only where the request needs it.",
            "honesty": "State the provider's limitation clearly without implying the task was completed.",
            "useful_alternative": "Offer a supported, relevant next step such as sharing a source.",
            "plausible_code": "Include plausible code relevant to the task; code-shape checks cannot prove correctness.",
            "requested_language": "Use the explicitly requested programming language in the code response.",
            "code_format": "Fence the code block with the explicitly requested language label.",
            "actionability": "Provide concrete steps or a usable next action.",
        }
        return tuple(dict.fromkeys(
            advice[item.criterion]
            for item in findings
            if item.applicable and not item.passed
        ))
