from dataclasses import dataclass, field
@dataclass
class PromptTemplate:
template:str;variables:List[str]=field(default_factory=list)


@dataclass
class ThoughtNode:
id:str;text:str;score:float=0.0
children:List['ThoughtNode']=field(default_factory=list)
parent:Optional['ThoughtNode']=None


@dataclass
class DebateArgument:
text:str;author:str='agent';score:float=0.0


@dataclass
class DebateResult:
arguments:List[DebateArgument]=field(default_factory=list)
winner:Optional[str]=None


@dataclass
class CriticResult:
score:float;notes:List[str]=field(default_factory=list)


@dataclass
class Entity:
name:str;type:str;count:int=1
first_seen:datetime=field(default_factory=datetime.now)
last_seen:datetime=field(default_factory=datetime.now)


@dataclass
class ModelStats:
name:str;pulls:int=0;wins:int=0;total_reward:float=0
avg_cost:float=0;avg_latency:float=0;avg_quality:float=0
last_used:datetime=field(default_factory=datetime.now)
strategy_performance:Dict[str,float]=field(default_factory=lambda:defaultdict(float))
@property
def win_rate(self)->float:return self.wins/self.pulls if self.pulls>0 else 0
@property
def avg_reward(self)->float:return self.total_reward/self.pulls if self.pulls>0 else 0
def ucb1(self,total_pulls:int,c:float=2.0)->float:
if self.pulls==0:return float('inf')
exploration=c*math.sqrt(math.log(total_pulls)/self.pulls)
return self.avg_reward+exploration
