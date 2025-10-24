# core/enums.py
from enum import Enum


class Intent(str,Enum):
QUESTION="question";COMMAND="command";CREATIVE="creative"
TECHNICAL="technical";CASUAL="casual";EXPLANATION="explanation"
REASONING="reasoning";ANALYSIS="analysis"


class Complexity(str,Enum):
SIMPLE="simple";MEDIUM="medium";COMPLEX="complex";EXPERT="expert"


class Strategy(str,Enum):
DIRECT="direct";COT="chain_of_thought";REFLECTION="reflection"
ENSEMBLE="ensemble";ITERATIVE="iterative"


class ReasoningType(str,Enum):
DEDUCTIVE="deductive";INDUCTIVE="inductive";ABDUCTIVE="abductive"
ANALOGICAL="analogical";CAUSAL="causal"


class MemoryType(str,Enum):
EPISODIC="episodic";SEMANTIC="semantic";PROCEDURAL="procedural";WORKING="working"


class LearningMode(str,Enum):
EXPLORATION="exploration";EXPLOITATION="exploitation";BALANCED="balanced"


class ReasoningPath(str,Enum):
FORWARD="forward";BACKWARD="backward";BIDIRECTIONAL="bidirectional"
