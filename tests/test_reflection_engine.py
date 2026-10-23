import unittest

from models import ResponseStatus
from reflection_engine import ReflectionEngine


class ReflectionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ReflectionEngine()

    def test_empty_response_scores_zero_and_is_reported(self) -> None:
        report = self.engine.evaluate("What is 2 + 2?", "")

        self.assertEqual(report.score, 0)
        self.assertEqual(report.response_status, ResponseStatus.EMPTY)
        finding = next(item for item in report.findings if item.criterion == "non_empty")
        self.assertFalse(finding.passed)
        self.assertIn("empty", finding.explanation.lower())

    def test_unsupported_fallback_is_honest_but_does_not_complete_request(self) -> None:
        report = self.engine.evaluate(
            "Explain the latest discovery on Mars.",
            "The offline provider has limited knowledge and cannot reliably "
            "answer this request, and it does not search the internet.",
        )

        self.assertLess(report.score, 60)
        self.assertEqual(report.response_status, ResponseStatus.LIMITATION)
        completion = next(
            item for item in report.findings if item.criterion == "task_completion"
        )
        honesty = next(item for item in report.findings if item.criterion == "honesty")
        self.assertFalse(completion.passed)
        self.assertIn("remains unanswered", completion.explanation)
        self.assertTrue(honesty.passed)
        self.assertTrue(honesty.applicable)

    def test_relevant_answer_scores_better_than_unrelated_or_fallback(self) -> None:
        request = "What is 2 + 2?"
        answer = self.engine.evaluate(request, "2 + 2 = 4.")
        unrelated = self.engine.evaluate(request, "Blue is a color.")
        fallback = self.engine.evaluate(
            request,
            "The offline provider has limited knowledge and cannot reliably "
            "answer this request.",
        )

        self.assertGreater(answer.score, unrelated.score)
        self.assertGreater(answer.score, fallback.score)
        self.assertEqual(unrelated.response_status, ResponseStatus.IRRELEVANT)

    def test_quality_scoring_rewards_relevant_concrete_answer(self) -> None:
        report = self.engine.evaluate("What is 2 + 2?", "2 + 2 = 4.")

        self.assertEqual(report.score, 100)
        self.assertEqual(report.weaknesses, ())
        self.assertTrue(report.strengths)

    def test_weaknesses_and_suggestions_are_specific(self) -> None:
        report = self.engine.evaluate("What is 2 + 2?", "Maybe.")

        self.assertLess(report.score, 100)
        self.assertTrue(report.weaknesses)
        self.assertIn(
            "Make the connection to the user's request explicit.",
            report.suggestions,
        )

    def test_programming_response_checks_for_code(self) -> None:
        report = self.engine.evaluate(
            "Write Python code to add two numbers.", "Two numbers can be added."
        )

        code_finding = next(
            item for item in report.findings if item.criterion == "plausible_code"
        )
        self.assertTrue(code_finding.applicable)
        self.assertFalse(code_finding.passed)
        self.assertTrue(
            any("plausible code" in suggestion.lower() for suggestion in report.suggestions)
        )

    def test_python_only_response_is_severely_penalized_for_embedded_c_request(self) -> None:
        request = "write a code in embadded C i need hello world as output"
        report = self.engine.evaluate(
            request, '```python\nprint("Hello, world!")\n```'
        )

        language = next(
            item for item in report.findings
            if item.criterion == "requested_language"
        )
        self.assertFalse(language.passed)
        self.assertIn("explicitly requested programming language", language.explanation)
        self.assertLessEqual(report.score, 30)
        self.assertNotEqual(report.response_status, ResponseStatus.ANSWERED)

    def test_embedded_c_response_meets_requested_language_and_output(self) -> None:
        request = "write a code in embadded C i need hello world as output"
        response = (
            "```c\n#include <stdio.h>\n\nint main(void)\n{\n"
            '    printf("Hello, World!\\n");\n'
            "    return 0;\n}\n```"
        )
        report = self.engine.evaluate(request, response)

        language = next(
            item for item in report.findings
            if item.criterion == "requested_language"
        )
        self.assertTrue(language.passed)
        self.assertEqual(report.response_status, ResponseStatus.ANSWERED)
        self.assertGreaterEqual(report.score, 80)

    def test_explicit_language_requires_a_matching_code_fence(self) -> None:
        report = self.engine.evaluate(
            "Write Embedded C Hello World.",
            "```text\n#include <stdio.h>\nint main(void) { printf(\"Hello, World!\"); }\n```",
        )
        formatting = next(
            item for item in report.findings if item.criterion == "code_format"
        )
        self.assertFalse(formatting.passed)
        self.assertIn("requested language label", formatting.explanation)

    def test_markdown_fence_without_code_is_not_claimed_as_code(self) -> None:
        report = self.engine.evaluate(
            "Write a Python function to check whether a number is even or odd.",
            "Here is the answer:\n```python\nThis function checks a number.\n```",
        )
        code_finding = next(
            item for item in report.findings if item.criterion == "plausible_code"
        )
        self.assertFalse(code_finding.passed)
        self.assertIn("No code example", report.weaknesses[0])

    def test_conversational_responses_are_evaluated_for_intent(self) -> None:
        generator_response = {
            "thanks": "You're welcome!",
            "bye": "Goodbye! Take care.",
            "how are you": "I'm here and ready to help. Thanks for asking!",
        }

        for request, response in generator_response.items():
            with self.subTest(request=request):
                report = self.engine.evaluate(request, response)
                self.assertGreaterEqual(report.score, 80)
                self.assertLessEqual(report.score, 100)
                self.assertTrue(
                    next(
                        finding
                        for finding in report.findings
                        if finding.criterion == "task_completion"
                    ).passed
                )

    def test_inappropriate_reply_to_greeting_is_not_scored_as_complete(self) -> None:
        report = self.engine.evaluate(
            "hello",
            "The offline provider has limited knowledge and cannot reliably "
            "answer this request.",
        )

        self.assertLess(report.score, 80)
        completion = next(
            finding
            for finding in report.findings
            if finding.criterion == "task_completion"
        )
        self.assertFalse(completion.passed)

    def test_greeting_is_high_but_not_assigned_a_perfect_score(self) -> None:
        from response_generator import ResponseGenerator

        response = ResponseGenerator().generate("hi")
        report = self.engine.evaluate("hi", response)

        self.assertGreaterEqual(report.score, 85)
        self.assertLess(report.score, 100)
        self.assertEqual(report.response_status, ResponseStatus.ANSWERED)

    def test_arithmetic_correctness_and_incorrect_attempt_are_distinguished(self) -> None:
        correct = self.engine.evaluate(
            "What is 25 multiplied by 4?", "25 * 4 = 100."
        )
        incorrect = self.engine.evaluate(
            "What is 25 multiplied by 4?", "25 * 4 = 99."
        )

        self.assertEqual(correct.findings[next(
            index for index, item in enumerate(correct.findings)
            if item.criterion == "factual_correctness"
        )].points_awarded, 25)
        self.assertGreater(correct.score, incorrect.score)
        self.assertTrue(any("does not match" in item for item in incorrect.weaknesses))

    def test_irrelevant_and_empty_answers_have_specific_weaknesses(self) -> None:
        irrelevant = self.engine.evaluate("What is 2 + 2?", "Blue is a color.")
        empty = self.engine.evaluate("Explain photosynthesis.", "  \n ")

        self.assertLess(irrelevant.score, 30)
        self.assertEqual(irrelevant.response_status, ResponseStatus.IRRELEVANT)
        self.assertTrue(any("relevant calculation" in item for item in irrelevant.weaknesses))
        self.assertEqual(empty.score, 0)
        self.assertTrue(any("no useful answer" in item.lower() for item in empty.weaknesses))

    def test_concise_correct_arithmetic_does_not_need_extra_detail(self) -> None:
        report = self.engine.evaluate("What is 2 + 2?", "4")

        self.assertGreaterEqual(report.score, 95)
        self.assertEqual(report.response_status, ResponseStatus.ANSWERED)
        self.assertFalse(any(item.criterion == "useful_detail" for item in report.findings))

    def test_actionability_is_only_evaluated_when_steps_are_requested(self) -> None:
        actionable = self.engine.evaluate(
            "How do I save a file?",
            "1. Choose Save from the menu.\n"
            "2. Select a location for the file.",
        )
        not_applicable = self.engine.evaluate("What is 2 + 2?", "4")

        self.assertTrue(any(
            item.criterion == "actionability" and item.passed
            for item in actionable.findings
        ))
        self.assertFalse(any(
            item.criterion == "actionability" for item in not_applicable.findings
        ))

    def test_general_answer_needs_request_relevance_not_just_readability(self) -> None:
        request = "Why do plants need sunlight?"
        relevant = self.engine.evaluate(
            request, "Plants use sunlight for photosynthesis."
        )
        unrelated = self.engine.evaluate(request, "Blue is a color.")

        self.assertGreater(relevant.score, unrelated.score)
        self.assertLess(relevant.score, 80)
        self.assertEqual(relevant.response_status, ResponseStatus.PARTIAL)
        self.assertEqual(unrelated.response_status, ResponseStatus.IRRELEVANT)

    def test_reflection_feedback_has_no_duplicate_items(self) -> None:
        report = self.engine.evaluate(
            "Write Python code to reverse a string.",
            "I cannot provide that. The offline provider has limited knowledge.",
        )

        self.assertEqual(len(report.strengths), len(set(report.strengths)))
        self.assertEqual(len(report.weaknesses), len(set(report.weaknesses)))
        self.assertEqual(len(report.suggestions), len(set(report.suggestions)))

    def test_cybersecurity_answer_is_relevant_actionable_and_not_claimed_verified(self) -> None:
        from response_generator import ResponseGenerator

        request = "Explain cybersecurity to a beginner."
        response = ResponseGenerator().generate(request)
        report = self.engine.evaluate(request, response)

        self.assertGreater(report.score, 70)
        self.assertEqual(report.response_status, ResponseStatus.ANSWERED)
        self.assertTrue(any(
            item.criterion == "task_completion" and item.passed
            for item in report.findings
        ))
        correctness = next(
            item for item in report.findings
            if item.criterion == "factual_correctness"
        )
        self.assertFalse(correctness.applicable)

    def test_missing_cybersecurity_details_are_specific(self) -> None:
        report = self.engine.evaluate(
            "How can I protect my computer?",
            "Keep software updated.",
        )

        self.assertTrue(any(
            "missing relevant points" in weakness for weakness in report.weaknesses
        ))
        self.assertTrue(report.suggestions)

    def test_identical_cybersecurity_evaluation_is_deterministic(self) -> None:
        request = "What is phishing?"
        response = "Phishing tricks people with fake messages to reveal information."

        first = self.engine.evaluate(request, response)
        second = self.engine.evaluate(request, response)

        self.assertEqual(first, second)
        self.assertTrue(0 <= first.score <= 100)


if __name__ == "__main__":
    unittest.main()
