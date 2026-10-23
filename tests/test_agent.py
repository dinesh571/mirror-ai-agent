import unittest
from unittest.mock import patch

from agent import MirrorAgent
from main import run_cli
from reflection_engine import ReflectionEngine
from response_generator import ResponseGenerator
from response_refiner import ResponseRefiner


class AgentTests(unittest.TestCase):
    def test_conversational_intents_return_contextual_responses(self) -> None:
        generator = ResponseGenerator()

        self.assertIn("Hello", generator.generate("hi"))
        self.assertIn("Hello", generator.generate("Hello"))
        self.assertIn("ready to help", generator.generate("how are you").lower())
        self.assertIn("welcome", generator.generate("thanks").lower())
        self.assertIn("goodbye", generator.generate("bye").lower())
        for greeting in ("hey", "good morning", "good afternoon", "good evening"):
            with self.subTest(greeting=greeting):
                self.assertIn("Hello", generator.generate(greeting))
        self.assertIn("welcome", generator.generate("thank you").lower())
        self.assertIn("take care", generator.generate("goodbye").lower())
        self.assertIn("take care", generator.generate("see you").lower())

    def test_required_conversation_inputs(self) -> None:
        generator = ResponseGenerator()

        for request, expected in (
            ("hi", "Hello"),
            ("hello", "Hello"),
            ("how are you?", "ready to help"),
            ("thanks", "welcome"),
            ("bye", "Goodbye"),
        ):
            with self.subTest(request=request):
                self.assertIn(expected.casefold(), generator.generate(request).casefold())

    def test_conversation_matching_is_case_insensitive_and_trims_whitespace(self) -> None:
        generator = ResponseGenerator()

        self.assertEqual(generator.generate("  HeLLo!  "), generator.generate("hello"))
        self.assertEqual(
            generator.generate("  GOOD MORNING  "),
            generator.generate("good morning"),
        )

    def test_unrelated_words_are_not_classified_as_conversation(self) -> None:
        generator = ResponseGenerator()

        with patch.object(generator, "_generate_with_ollama", return_value=""):
            self.assertIn("limited knowledge", generator.generate("The ship is nearby."))
            self.assertIn(
                "limited knowledge",
                generator.generate("A new beginning is possible."),
            )

    def test_greeting_reply_scores_higher_than_generic_fallback(self) -> None:
        engine = ReflectionEngine()
        greeting = engine.evaluate("hi", ResponseGenerator().generate("hi"))
        fallback = engine.evaluate(
            "hi",
            "The offline provider has limited knowledge and cannot reliably "
            "answer this request.",
        )

        self.assertGreater(greeting.score, fallback.score)
        self.assertGreaterEqual(greeting.score, 80)
        self.assertLessEqual(greeting.score, 100)

    def test_end_to_end_arithmetic_workflow(self) -> None:
        result = MirrorAgent().run("What is 7 * 6?")

        self.assertIn("7 * 6 = 42", result.initial_response)
        self.assertEqual(result.reflection.score, 100)
        self.assertEqual(result.refined_response, result.initial_response)
        self.assertEqual(result.final_reflection.score, 100)
        self.assertIn("preserved", result.refinement_note)

    def test_multiplied_by_request_returns_correct_arithmetic(self) -> None:
        result = MirrorAgent().run("What is 25 multiplied by 4?")

        self.assertIn("100", result.initial_response)
        self.assertGreaterEqual(result.reflection.score, 95)
        self.assertEqual(
            result.final_reflection.score,
            ReflectionEngine().evaluate(
                result.request, result.refined_response
            ).score,
        )

    def test_all_arithmetic_operations_and_division_by_zero(self) -> None:
        generator = ResponseGenerator()
        cases = (
            ("What is 9 plus 3?", "9 + 3 = 12"),
            ("What is 9 minus 3?", "9 - 3 = 6"),
            ("What is 9 times 3?", "9 * 3 = 27"),
            ("What is 9 divided by 3?", "9 / 3 = 3"),
        )
        for request, answer in cases:
            with self.subTest(request=request):
                self.assertIn(answer, generator.generate(request))

        self.assertIn(
            "undefined",
            generator.generate("What is 9 divided by 0?").casefold(),
        )

    def test_python_programming_request_returns_relevant_example(self) -> None:
        result = MirrorAgent().run(
            "Write a Python program to add two numbers."
        )

        self.assertIn("def add_numbers", result.initial_response)
        self.assertEqual(result.reflection.response_status.value, "answered")
        self.assertGreaterEqual(result.reflection.score, 80)

    def test_supported_python_examples(self) -> None:
        cases = (
            ("Write a Python function to check whether a number is even or odd", "even_or_odd"),
            ("Write a Python program to reverse a string", "reverse_string"),
            ("Find the largest number in a Python list", "largest_number"),
        )
        generator = ResponseGenerator()
        engine = ReflectionEngine()
        for request, code_name in cases:
            with self.subTest(request=request):
                response = generator.generate(request)
                self.assertIn(code_name, response)
                self.assertTrue(engine._has_plausible_code(response))
                self.assertNotIn("limited knowledge", response)

    def test_explicit_language_hello_world_requests_keep_the_requested_language(self) -> None:
        cases = (
            ("Write Python Hello World.", "```python", "print("),
            ("Write a C Hello World program.", "```c", "printf("),
            ("Write a C++ Hello World program.", "```cpp", "std::cout"),
            ("Write a Java Hello World program.", "```java", "System.out.println"),
            ("Write a JavaScript Hello World program.", "```javascript", "console.log"),
        )
        generator = ResponseGenerator()
        for request, fence, code in cases:
            with self.subTest(request=request):
                with patch.object(generator, "_generate_with_ollama") as inference:
                    response = generator.generate(request)
                self.assertIn(fence, response)
                self.assertIn(code, response)
                inference.assert_not_called()

    def test_embedded_c_spelling_variations_are_supported(self) -> None:
        generator = ResponseGenerator()
        for spelling in ("embedded C", "embadded C", "embeded C"):
            request = f"Write a code in {spelling}; I need Hello World as output."
            with self.subTest(spelling=spelling):
                response = generator.generate(request)
                self.assertIn("```c", response)
                self.assertIn("#include <stdio.h>", response)
                self.assertIn('printf("Hello, World!', response)
                self.assertNotIn("```python", response)

    def test_embedded_platform_request_is_left_for_model_specific_guidance(self) -> None:
        generator = ResponseGenerator()
        with patch.object(
            generator, "_generate_with_ollama", return_value="Platform-specific firmware."
        ) as inference:
            response = generator.generate(
                "Write Embedded C for an STM32 that prints Hello World."
            )
        self.assertEqual(response, "Platform-specific firmware.")
        inference.assert_called_once()

    def test_agi_has_a_concise_non_repetitive_reference_answer(self) -> None:
        response = ResponseGenerator().generate("ok what is AGI")
        lowered = response.casefold()
        self.assertIn("artificial general intelligence", lowered)
        self.assertIn("broad range", lowered)
        self.assertNotIn("general ai means", lowered)
        self.assertGreater(
            ReflectionEngine().evaluate("What is AGI?", response).score, 60
        )

    def test_refiner_preserves_a_correct_embedded_c_answer(self) -> None:
        request = "write a code in embadded C i need hello world as output"
        engine = ReflectionEngine()
        generator = ResponseGenerator()
        response = generator.generate(request)
        report = engine.evaluate(request, response)
        from response_refiner import ResponseRefiner

        result = ResponseRefiner(generator, engine).refine(request, response, report)
        self.assertEqual(result.response, response)
        self.assertFalse(result.changed)
        self.assertIn("```c", result.response)

    def test_supported_general_knowledge_topics(self) -> None:
        cases = (
            ("What is AI?", "artificial intelligence"),
            ("What is machine learning?", "machine learning"),
            ("What is the difference between AI and machine learning?", "broader"),
            ("What is Python?", "programming language"),
            ("What is a database?", "organized collection"),
            ("What does an operating system do?", "hardware"),
            ("Explain the difference between a Python list and tuple.", "mutable"),
        )
        generator = ResponseGenerator()
        engine = ReflectionEngine()
        for request, key_phrase in cases:
            with self.subTest(request=request):
                response = generator.generate(request)
                self.assertIn(key_phrase, response.casefold())
                self.assertNotIn("limited knowledge", response.casefold())
                self.assertGreater(engine.evaluate(request, response).score, 50)

    def test_unsupported_request_has_limitation_and_relevant_next_step(self) -> None:
        generator = ResponseGenerator()
        with patch.object(generator, "_generate_with_ollama", return_value=""):
            response = generator.generate("Explain quantum archaeology.")

        self.assertIn("limited knowledge", response)
        self.assertIn("share a source", response)

    def test_supported_cybersecurity_responses(self) -> None:
        cases = (
            ("Explain cybersecurity to a beginner.", ("cybersecurity", "password", "risk")),
            ("What is cybersecurity?", ("protecting", "digital attacks")),
            ("How can I protect my computer?", ("updates", "back up")),
            ("What is phishing?", ("trick", "trusted")),
            ("How can I create a strong password?", ("long", "unique")),
            ("What is two-factor authentication?", ("second", "password")),
        )
        generator = ResponseGenerator()
        for request, required_terms in cases:
            with self.subTest(request=request):
                response = generator.generate(request)
                for term in required_terms:
                    self.assertIn(term, response.casefold())
                self.assertNotIn("limited knowledge", response.casefold())

    def test_unrelated_request_is_not_routed_to_cybersecurity(self) -> None:
        generator = ResponseGenerator()
        with patch.object(generator, "_generate_with_ollama", return_value=""):
            response = generator.generate(
                "What is the best way to organize my computer desktop?"
            )

        self.assertNotIn("phishing", response.casefold())
        self.assertNotIn("two-factor", response.casefold())
        self.assertIn("limited knowledge", response.casefold())

    def test_cybersecurity_refinement_has_stable_scores(self) -> None:
        agent = MirrorAgent()
        result = agent.run("Explain cybersecurity to a beginner.")
        report_again = ReflectionEngine().evaluate(
            result.request, result.refined_response
        )
        repeated = ReflectionEngine().evaluate(
            result.request, result.refined_response
        )

        self.assertIn("Cybersecurity is the practice", result.initial_response)
        self.assertEqual(result.initial_response, result.refined_response)
        self.assertEqual(result.reflection.score, result.final_reflection.score)
        self.assertEqual(result.final_reflection.score, report_again.score)
        self.assertEqual(report_again, repeated)

    def test_unchanged_unsupported_response_keeps_same_score_and_status_once(self) -> None:
        request = "Explain quantum archaeology."
        with patch.object(ResponseGenerator, "_generate_with_ollama", return_value=""):
            result = MirrorAgent().run(request)
        direct = ReflectionEngine().evaluate(request, result.refined_response)

        self.assertEqual(result.initial_response, result.refined_response)
        self.assertEqual(result.reflection.score, result.final_reflection.score)
        self.assertEqual(result.final_reflection.score, direct.score)
        self.assertEqual(result.refinement_note.count("preserved"), 1)

    def test_cli_displays_refinement_status_separately_once(self) -> None:
        outputs = []
        responses = iter(["Explain cybersecurity to a beginner.", "exit"])
        run_cli(lambda _prompt: next(responses), outputs.append)

        self.assertEqual(
            sum("Refinement status:" in item for item in outputs), 1
        )
        self.assertEqual(
            sum("Cybersecurity is the practice" in item for item in outputs), 1
        )
        self.assertFalse(any(
            "No candidate improved" in item and "Cybersecurity is the practice" in item
            for item in outputs
        ))

    def test_cli_shows_unchanged_fallback_and_refinement_status_once(self) -> None:
        outputs = []
        responses = iter(["Explain quantum archaeology.", "exit"])
        with patch.object(ResponseGenerator, "_generate_with_ollama", return_value=""):
            run_cli(lambda _prompt: next(responses), outputs.append)
        combined_output = "\n".join(outputs)
        score_lines = [
            line for line in outputs
            if "Reflection report (" in line or "Final evaluation score:" in line
        ]

        self.assertEqual(combined_output.count("limited knowledge"), 1)
        self.assertEqual(combined_output.count("No candidate improved"), 1)
        self.assertEqual(len(score_lines), 2)
        self.assertIn("score: 46/100", score_lines[0])
        self.assertIn("46/100", score_lines[1])

    def test_weak_supported_answer_is_refined_from_offline_template(self) -> None:
        generator = ResponseGenerator()
        engine = ReflectionEngine()
        refiner = ResponseRefiner(generator, engine)
        weak_response = "Maybe use a function."
        report = engine.evaluate("Write Python code to add two numbers.", weak_response)

        refined = refiner.refine(
            "Write Python code to add two numbers.", weak_response, report
        )

        self.assertTrue(refined.changed)
        self.assertIn("def add_numbers", refined.response)
        self.assertGreater(
            engine.evaluate(
                "Write Python code to add two numbers.", refined.response
            ).score,
            report.score,
        )

    def test_unsupported_request_is_transparent(self) -> None:
        with patch.object(ResponseGenerator, "_generate_with_ollama", return_value=""):
            result = MirrorAgent().run("What are the latest scientific discoveries?")

        self.assertIn("can't verify", result.initial_response)
        self.assertIn("offline provider", result.initial_response)
        self.assertIn("share a relevant article", result.initial_response)
        self.assertEqual(result.refined_response, result.initial_response)
        self.assertLess(result.reflection.score, 60)
        self.assertEqual(result.reflection.response_status.value, "limitation")
        self.assertEqual(
            result.final_reflection.score,
            ReflectionEngine().evaluate(
                result.request, result.refined_response
            ).score,
        )

    def test_refinement_improves_wrong_arithmetic_without_changing_correct_answer(self) -> None:
        engine = ReflectionEngine()
        refiner = ResponseRefiner(ResponseGenerator(), engine)
        request = "What is 2 + 2?"
        correct = "2 + 2 = 4."
        correct_report = engine.evaluate(request, correct)
        preserved = refiner.refine(request, correct, correct_report)
        wrong = "2 + 2 = 5."
        wrong_report = engine.evaluate(request, wrong)
        improved = refiner.refine(request, wrong, wrong_report)

        self.assertEqual(preserved.response, correct)
        self.assertFalse(preserved.changed)
        self.assertEqual(improved.response, "2 + 2 = 4.")
        self.assertTrue(improved.changed)
        self.assertGreater(
            engine.evaluate(request, improved.response).score,
            wrong_report.score,
        )

    def test_empty_response_gets_a_clear_refinement(self) -> None:
        engine = ReflectionEngine()
        refiner = ResponseRefiner(ResponseGenerator(), engine)
        report = engine.evaluate("What is 2 + 2?", "")

        refined = refiner.refine("What is 2 + 2?", "", report)

        self.assertTrue(refined.changed)
        self.assertIn("2 + 2 = 4", refined.response)

    def test_cli_handles_empty_input_and_ctrl_c(self) -> None:
        outputs = []
        responses = iter(["  ", "What is 2 + 2?"])

        def input_then_interrupt(_prompt: str) -> str:
            try:
                return next(responses)
            except StopIteration:
                raise KeyboardInterrupt

        run_cli(input_then_interrupt, outputs.append)

        self.assertTrue(any("non-empty" in item for item in outputs))
        self.assertTrue(any("Initial response:" in item for item in outputs))
        self.assertEqual(outputs[-1], "\nGoodbye.")


if __name__ == "__main__":
    unittest.main()
