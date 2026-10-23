"""Interactive command-line interface for MIRROR."""

from typing import Callable

from agent import MirrorAgent
from models import AgentResult


BANNER = r"""
 __  __ ___ ____  ____   ___  ____
|  \/  |_ _|  _ \|  _ \ / _ \|  _ \
| |\/| || || |_) | |_) | | | | |_) |
| |  | || ||  _ <|  _ <| |_| |  _ <
|_|  |_|___|_| \_\_| \_\\___/|_| \_\
             AI AGENT v0.1
"""


def _display_result(result: AgentResult, output_fn: Callable[[str], None]) -> None:
    output_fn("\nInitial response:")
    output_fn(result.initial_response or "(empty response)")
    output_fn(
        f"\nReflection report ({result.reflection.response_status.value}, "
        f"score: {result.reflection.score}/100):"
    )
    output_fn("What the response does well:")
    _display_list(result.reflection.strengths, output_fn)
    output_fn("Identified weaknesses:")
    _display_list(result.reflection.weaknesses, output_fn)
    output_fn("Improvement suggestions:")
    _display_list(result.reflection.suggestions, output_fn)
    output_fn("\nRefined response:")
    if result.refined_response == result.initial_response:
        output_fn("(Unchanged from the initial response.)")
    else:
        output_fn(result.refined_response)
    output_fn("Refinement status:")
    output_fn(result.refinement_note)
    output_fn(
        f"\nFinal evaluation score: {result.final_reflection.score}/100"
    )


def _display_list(items: tuple[str, ...], output_fn: Callable[[str], None]) -> None:
    if not items:
        output_fn("- None identified.")
        return
    for item in items:
        output_fn(f"- {item}")


def run_cli(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    agent = MirrorAgent()
    output_fn(BANNER)
    output_fn("Enter a question or task. Type 'exit' to quit.")
    while True:
        try:
            request = input_fn("\nMIRROR> ")
            if request.strip().lower() in {"exit", "quit", "q"}:
                output_fn("Goodbye.")
                return
            if not request.strip():
                output_fn("Please enter a non-empty question or task.")
                continue

            _display_result(agent.run(request.strip()), output_fn)
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye.")
            return


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
