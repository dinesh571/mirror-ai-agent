import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from agent import MirrorAgent
from reflection_engine import ReflectionEngine
from models import ProgrammingLanguage
from response_generator import ResponseGenerator, detect_programming_language


class ResponseGeneratorOllamaTests(unittest.TestCase):
    def test_sql_is_detected_as_an_explicit_language_without_incidental_matches(self) -> None:
        self.assertEqual(
            detect_programming_language("Write an SQL query to select all users."),
            ProgrammingLanguage.SQL,
        )
        self.assertIsNone(detect_programming_language("The ship is near the sea."))

    def test_untemplated_explicit_language_uses_ollama_and_remains_evaluable(self) -> None:
        response = "```sql\nSELECT name FROM users;\n```"
        with patch.object(
            ResponseGenerator, "_generate_with_ollama", return_value=response
        ) as inference:
            generated = ResponseGenerator().generate(
                "Write an SQL query to select all user names."
            )

        inference.assert_called_once()
        self.assertEqual(generated, response)
        report = ReflectionEngine().evaluate(
            "Write an SQL query to select all user names.", generated
        )
        self.assertTrue(next(
            item for item in report.findings
            if item.criterion == "requested_language"
        ).passed)

    def test_deterministic_handlers_do_not_call_ollama(self) -> None:
        generator = ResponseGenerator()
        with patch.object(generator, "_generate_with_ollama") as ollama_call:
            self.assertEqual(generator.generate("hi"), "Hello! How can I help you today?")
            self.assertEqual(generator.generate("What is 9 divided by 3?"), "9 / 3 = 3.")
            self.assertIn("def add_numbers", generator.generate(
                "Write a Python program to add two numbers."
            ))
        ollama_call.assert_not_called()

    def test_general_request_uses_configured_model_and_system_prompt(self) -> None:
        client = Mock()
        client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="Gravity attracts objects with mass.")
        )
        ollama_module = ModuleType("ollama")
        ollama_module.Client = Mock(return_value=client)

        with (
            patch.dict(os.environ, {"MIRROR_OLLAMA_MODEL": "custom:qwen"}),
            patch.dict(sys.modules, {"ollama": ollama_module}),
        ):
            generator = ResponseGenerator()
            response = generator.generate("Explain gravity simply.")

        self.assertEqual(response, "Gravity attracts objects with mass.")
        ollama_module.Client.assert_called_once_with(timeout=45.0)
        client.chat.assert_called_once()
        kwargs = client.chat.call_args.kwargs
        self.assertEqual(kwargs["model"], "custom:qwen")
        self.assertEqual(kwargs["messages"][-1], {
            "role": "user",
            "content": "Explain gravity simply.",
        })
        self.assertIn("Acknowledge uncertainty", kwargs["messages"][0]["content"])
        self.assertFalse(kwargs["stream"])

    def test_default_model_name_is_used(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            generator = ResponseGenerator()
        self.assertEqual(generator.model_name, "qwen2.5:3b")

    def test_ollama_failure_returns_limitation_string_without_crashing(self) -> None:
        client = Mock()
        client.chat.side_effect = TimeoutError("local inference timed out")
        ollama_module = ModuleType("ollama")
        ollama_module.Client = Mock(return_value=client)

        with patch.dict(sys.modules, {"ollama": ollama_module}):
            response = ResponseGenerator().generate("Explain gravity.")

        self.assertIsInstance(response, str)
        self.assertIn("limited knowledge", response)

    def test_missing_ollama_package_returns_limitation_string(self) -> None:
        with patch.dict(sys.modules, {"ollama": None}):
            response = ResponseGenerator().generate("Explain gravity.")

        self.assertIsInstance(response, str)
        self.assertIn("limited knowledge", response)

    def test_empty_model_response_returns_limitation_string(self) -> None:
        client = Mock()
        client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="   ")
        )
        ollama_module = ModuleType("ollama")
        ollama_module.Client = Mock(return_value=client)

        with patch.dict(sys.modules, {"ollama": ollama_module}):
            response = ResponseGenerator().generate("Explain gravity.")

        self.assertIn("limited knowledge", response)

    def test_model_generated_response_is_evaluated_and_agent_calls_once(self) -> None:
        response = "Gravity is the attraction between objects with mass."
        with patch.object(
            ResponseGenerator, "_generate_with_ollama", return_value=response
        ) as inference:
            result = MirrorAgent().run("Explain gravity.")

        inference.assert_called_once_with("Explain gravity.")
        self.assertEqual(result.initial_response, response)
        self.assertEqual(result.refined_response, response)
        self.assertEqual(
            result.final_reflection.score,
            ReflectionEngine().evaluate(
                result.request, result.refined_response
            ).score,
        )

    def test_refiner_does_not_run_second_model_inference(self) -> None:
        response = "Gravity pulls objects toward one another."
        generator = ResponseGenerator()
        with patch.object(
            generator, "_generate_with_ollama", return_value=response
        ) as inference:
            first = generator.generate("Explain gravity.")
            report = ReflectionEngine().evaluate("Explain gravity.", first)
            from response_refiner import ResponseRefiner

            result = ResponseRefiner(
                generator, ReflectionEngine()
            ).refine("Explain gravity.", first, report)

        inference.assert_called_once_with("Explain gravity.")
        self.assertEqual(result.response, first)


if __name__ == "__main__":
    unittest.main()
