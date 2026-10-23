# MIRROR — Reflection Agent v0.1

MIRROR is a small command-line reflection agent. It first uses deterministic
offline handlers, then can use a locally running Ollama model for requests not
covered by those handlers. Responses are checked with deterministic rules and
passed through a conservative refinement stage. MIRROR does not use a cloud AI
service or execute generated code.

## How reflection works

The evaluator selects deterministic checks by intent. It verifies results for
supported arithmetic, checks programming answers for relevant code shape and
explicit language/format requirements, and checks conversational replies
against the recognized conversational intent.
For unsupported current-information requests, it checks whether the limitation
and a useful next step are stated while keeping task completion unfulfilled.
Clarity, relevant detail, and actionability are scored only when appropriate.
Reports include weighted findings, strengths, weaknesses, improvement
suggestions, and a score from 0 to 100. General-language relevance is a
keyword heuristic; code-shape checks do not prove code correctness, and no
rule-based check establishes general factual correctness.

The refiner uses only supported offline examples and presentation changes. If
it cannot make a meaningful supported improvement, it keeps the original
answer and says why.

## Architecture

- `main.py` — interactive CLI and output formatting.
- `agent.py` — coordinates generation, reflection, refinement, and final scoring.
- `response_generator.py` — deterministic arithmetic, conversational replies,
  selected Python examples, basic Hello World examples for common requested
  languages, a small general-knowledge catalog, beginner-level cybersecurity
  explanations, and a local Ollama fallback.
- `reflection_engine.py` — structured, rule-based evaluation, including
  explicit programming-language mismatch checks.
- `response_refiner.py` — applies safe, supported improvements.
- `models.py` — typed dataclasses and request categories.
- `tests/` — deterministic tests using the standard-library `unittest` and
  mocked Ollama responses.

## Windows setup

Install the Python Ollama client into the existing environment if it is not
already installed:

```powershell
python -m pip install -r requirements.txt
```

Ollama must be running locally with the selected model already available. The
default model is `qwen2.5:3b`. You may select another installed model using:

```powershell
$env:MIRROR_OLLAMA_MODEL = "qwen2.5:3b"
```

From PowerShell in the project directory, activate the existing environment:

```powershell
.\mirenv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\mirenv\Scripts\Activate.ps1
```

Alternatively, invoke the environment's Python directly without activation:

```powershell
.\mirenv\Scripts\python.exe main.py
```

## Run MIRROR

```powershell
python main.py
```

Enter a task at the prompt. Type `exit`, `quit`, or `q` to leave.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Current limitations

Deterministic handlers cover simple arithmetic, selected Python examples,
basic Hello World examples in selected languages, common conversational inputs,
cybersecurity basics, and a small set of general knowledge topics. Other
requests use the local Ollama model when available and fall back to an explicit
limitation if the package, service, model, or response is unavailable.
Programming-language and code-format checks are heuristic: they can flag clear
mismatches, but do not establish code correctness. General relevance and
knowledge checks also cannot verify every factual claim. Model responses are
not proof of factual correctness, and generated code is never executed.
