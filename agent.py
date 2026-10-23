"""Orchestrates generation, reflection, and conservative refinement."""

from models import AgentResult
from reflection_engine import ReflectionEngine
from response_generator import ResponseGenerator
from response_refiner import ResponseRefiner


class MirrorAgent:
    def __init__(self) -> None:
        self._generator = ResponseGenerator()
        self._reflection_engine = ReflectionEngine()
        self._refiner = ResponseRefiner(
            self._generator, self._reflection_engine
        )

    def run(self, request: str) -> AgentResult:
        initial_response = self._generator.generate(request)
        reflection = self._reflection_engine.evaluate(request, initial_response)
        refinement = self._refiner.refine(
            request, initial_response, reflection
        )
        final_reflection = self._reflection_engine.evaluate(
            request, refinement.response
        )
        return AgentResult(
            request=request,
            initial_response=initial_response,
            reflection=reflection,
            refined_response=refinement.response,
            final_reflection=final_reflection,
            refinement_note=refinement.explanation,
        )
