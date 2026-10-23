"""Small data structures shared by the MIRROR agent."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class RequestKind(Enum):
    ARITHMETIC = "arithmetic"
    PYTHON = "python"
    GENERAL = "general"


class ProgrammingLanguage(Enum):
    EMBEDDED_C = "embedded_c"
    C = "c"
    CPP = "cpp"
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    SQL = "sql"


class ResponseStatus(Enum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    LIMITATION = "limitation"
    EMPTY = "empty"
    IRRELEVANT = "irrelevant"


class ConversationIntent(Enum):
    GREETING = "greeting"
    WELL_BEING = "well_being"
    GRATITUDE = "gratitude"
    FAREWELL = "farewell"


class KnowledgeIntent(Enum):
    AGI = "agi"
    AI_VS_ML = "ai_vs_ml"
    MACHINE_LEARNING = "machine_learning"
    AI = "ai"
    PYTHON = "python"
    DATABASE = "database"
    OPERATING_SYSTEM = "operating_system"
    PYTHON_LIST_TUPLE = "python_list_tuple"


class CybersecurityIntent(Enum):
    INTRODUCTION = "introduction"
    PROTECT_COMPUTER = "protect_computer"
    PHISHING = "phishing"
    STRONG_PASSWORD = "strong_password"
    TWO_FACTOR = "two_factor"


@dataclass(frozen=True)
class EvaluationFinding:
    criterion: str
    passed: bool
    explanation: str
    deduction: int = 0
    applicable: bool = True
    weight: int = 0
    points_awarded: int = 0


@dataclass(frozen=True)
class ReflectionReport:
    score: int
    response_status: ResponseStatus
    findings: Tuple[EvaluationFinding, ...]
    strengths: Tuple[str, ...]
    weaknesses: Tuple[str, ...]
    suggestions: Tuple[str, ...]


@dataclass(frozen=True)
class RefinementResult:
    response: str
    changed: bool
    explanation: str


@dataclass(frozen=True)
class AgentResult:
    request: str
    initial_response: str
    reflection: ReflectionReport
    refined_response: str
    final_reflection: ReflectionReport
    refinement_note: str
