#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Advanced Self-Adapting Enterprise LLM Chat System
思考の連鎖・自己反省・知識グラフ・A/Bテスト・アンサンブル学習

Features:
- Chain-of-Thought reasoning with self-reflection
- Knowledge Graph with relation extraction
- Multi-strategy ensemble learning
- A/B testing framework with statistical analysis
- Contextual bandits with Thompson Sampling
- Automated prompt engineering & evolution
- Meta-cognitive monitoring
- Adversarial robustness checking
- Uncertainty quantification

使い方:
1. export GROQ_API_KEY='your_key'
2. pip install groq numpy
3. python ultra-llm.py
"""

import os,sys,time,json,hashlib,logging,asyncio,re,uuid,math,statistics
from typing import Optional,List,Dict,Any,Callable,Tuple,Set,Union
from dataclasses import dataclass,field,asdict
from collections import defaultdict,Counter,deque
from datetime import datetime,timedelta
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed

try:
    from groq import Groq,RateLimitError,APIError
    GROQ_OK = True
except ImportError:
    print("❌ pip install groq numpy");sys.exit(1)

try:import readline
except:pass

# ========== 列挙型 ==========
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

# ========== 設定 ==========
@dataclass
class Cfg:
    model:str="llama-3.1-8b-instant";max_tok:int=4000;temp:float=0.7
    vec_db:bool=True;dim:int=384;ttl:int=3600;sim:float=0.92
    retry:int=3;delay:float=1.0;max_len:int=10000
    # Advanced features
    adapt:bool=True;mab:bool=True;memory:bool=True
    cot:bool=True;reflection:bool=True;kg:bool=True
    ab_test:bool=True;ensemble:bool=True;metacog:bool=True
    thompson:bool=True;adversarial:bool=False

# ========== データ構造 ==========
@dataclass
class Resp:
    text:str;conf:float;tok:int=0;pt:int=0;ct:int=0;lat:float=0
    cost:float=0;model:str="";ts:datetime=field(default_factory=datetime.now)
    reason:str="unknown";cache:bool=False;sim:float=0;rating:Optional[int]=None
    intent:Optional[Intent]=None;complexity:Optional[Complexity]=None
    sentiment:float=0;strategy:Optional[Strategy]=None
    reasoning_steps:List[str]=field(default_factory=list)
    reflection:Optional[str]=None;uncertainty:float=0;alternatives:List[str]=field(default_factory=list)
    
    @property
    def ok(self)->bool:return self.reason in("stop","length")
    
    def dict(self)->Dict:
        return{
            'text':self.text,'conf':self.conf,'tok':self.tok,'cost':self.cost,
            'model':self.model,'lat':self.lat,'ok':self.ok,'cache':self.cache,
            'rating':self.rating,'intent':self.intent,'complexity':self.complexity,
            'sentiment':self.sentiment,'strategy':self.strategy,
            'reasoning_steps':self.reasoning_steps,'reflection':self.reflection,
            'uncertainty':self.uncertainty,'alternatives':self.alternatives
        }

@dataclass
class KnowledgeNode:
    """知識グラフのノード"""
    id:str;name:str;type:str;properties:Dict[str,Any]=field(default_factory=dict)
    created:datetime=field(default_factory=datetime.now)
    confidence:float=1.0;sources:List[str]=field(default_factory=list)

@dataclass
class KnowledgeEdge:
    """知識グラフのエッジ(関係)"""
    source:str;target:str;relation:str;weight:float=1.0
    properties:Dict[str,Any]=field(default_factory=dict)
    created:datetime=field(default_factory=datetime.now)

@dataclass
class KnowledgeGraph:
    """知識グラフ"""
    nodes:Dict[str,KnowledgeNode]=field(default_factory=dict)
    edges:List[KnowledgeEdge]=field(default_factory=list)
    
    def add_node(self,node:KnowledgeNode):
        self.nodes[node.id]=node
    
    def add_edge(self,edge:KnowledgeEdge):
        self.edges.append(edge)
    
    def get_neighbors(self,node_id:str,relation:Optional[str]=None)->List[str]:
        """隣接ノードを取得"""
        neighbors=[]
        for e in self.edges:
            if e.source==node_id and(relation is None or e.relation==relation):
                neighbors.append(e.target)
            elif e.target==node_id and(relation is None or e.relation==relation):
                neighbors.append(e.source)
        return neighbors
    
    def get_path(self,start:str,end:str,max_depth:int=3)->Optional[List[str]]:
        """2ノード間のパスを探索(BFS)"""
        if start not in self.nodes or end not in self.nodes:return None
        
        queue=deque([(start,[start])])
        visited=set([start])
        
        while queue:
            current,path=queue.popleft()
            if len(path)>max_depth:continue
            if current==end:return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor,path+[neighbor]))
        return None
    
    def get_related_concepts(self,concept:str,top_k:int=5)->List[Tuple[str,float]]:
        """関連概念を取得"""
        if concept not in self.nodes:return[]
        
        # 直接の隣接ノード
        neighbors=self.get_neighbors(concept)
        
        # 重みでソート
        weighted=[]
        for n in neighbors:
            edges=[e for e in self.edges if(e.source==concept and e.target==n)or(e.target==concept and e.source==n)]
            if edges:weighted.append((n,max(e.weight for e in edges)))
        
        weighted.sort(key=lambda x:x[1],reverse=True)
        return weighted[:top_k]

@dataclass
class ABTest:
    """A/Bテスト実験"""
    name:str;variant_a:Dict[str,Any];variant_b:Dict[str,Any]
    results_a:List[float]=field(default_factory=list)
    results_b:List[float]=field(default_factory=list)
    started:datetime=field(default_factory=datetime.now)
    
    def add_result(self,variant:str,value:float):
        if variant=='A':self.results_a.append(value)
        else:self.results_b.append(value)
    
    def get_winner(self,confidence:float=0.95)->Optional[str]:
        """統計的に有意な勝者を判定"""
        if len(self.results_a)<10 or len(self.results_b)<10:return None
        
        mean_a=statistics.mean(self.results_a)
        mean_b=statistics.mean(self.results_b)
        
        # 簡易的なt検定(正規分布を仮定)
        if len(self.results_a)>1 and len(self.results_b)>1:
            var_a=statistics.variance(self.results_a)
            var_b=statistics.variance(self.results_b)
            n_a,n_b=len(self.results_a),len(self.results_b)
            
            # 標準誤差
            se=math.sqrt(var_a/n_a+var_b/n_b)
            if se>0:
                t_stat=abs(mean_a-mean_b)/se
                # 簡易判定(t>2で有意)
                if t_stat>2.0:
                    return'A'if mean_a>mean_b else'B'
        
        return None

@dataclass
class ThompsonSampler:
    """Thompson Sampling for Contextual Bandits"""
    actions:List[str]
    alpha:Dict[str,List[float]]=field(default_factory=dict)  # 成功
    beta:Dict[str,List[float]]=field(default_factory=dict)   # 失敗
    
    def __post_init__(self):
        for action in self.actions:
            self.alpha[action]=[1.0]
            self.beta[action]=[1.0]
    
    def select(self,context:Optional[str]=None)->str:
        """Bayesian最適化でアクションを選択"""
        samples={}
        for action in self.actions:
            # Beta分布からサンプリング
            a=sum(self.alpha[action])
            b=sum(self.beta[action])
            samples[action]=np.random.beta(a,b)
        return max(samples.items(),key=lambda x:x[1])[0]
    
    def update(self,action:str,reward:float):
        """報酬でBeta分布を更新"""
        if reward>0.5:
            self.alpha[action].append(reward)
        else:
            self.beta[action].append(1-reward)
        
        # メモリ管理
        if len(self.alpha[action])>100:
            self.alpha[action]=self.alpha[action][-100:]
            self.beta[action]=self.beta[action][-100:]

@dataclass
class MetaCognition:
    """メタ認知モニタリング"""
    confidence_history:List[float]=field(default_factory=list)
    uncertainty_history:List[float]=field(default_factory=list)
    error_patterns:Dict[str,int]=field(default_factory=lambda:defaultdict(int))
    
    def add_result(self,confidence:float,uncertainty:float,correct:bool,error_type:Optional[str]=None):
        self.confidence_history.append(confidence)
        self.uncertainty_history.append(uncertainty)
        if not correct and error_type:
            self.error_patterns[error_type]+=1
    
    def get_calibration_score(self)->float:
        """信頼度のキャリブレーションスコア"""
        if len(self.confidence_history)<10:return 0.5
        return statistics.mean(self.confidence_history)
    
    def should_seek_clarification(self)->bool:
        """明確化が必要かを判定"""
        if len(self.uncertainty_history)<3:return False
        recent=self.uncertainty_history[-3:]
        return statistics.mean(recent)>0.7
    
    def get_weakness_areas(self,top_k:int=3)->List[str]:
        """弱点領域を特定"""
        return[k for k,v in sorted(self.error_patterns.items(),
               key=lambda x:x[1],reverse=True)[:top_k]]

@dataclass
class PromptTemplate:
    """進化するプロンプトテンプレート"""
    id:str;template:str;category:str
    usage_count:int=0;success_count:int=0
    avg_quality:float=0.5;created:datetime=field(default_factory=datetime.now)
    mutations:int=0
    
    @property
    def success_rate(self)->float:
        return self.success_count/self.usage_count if self.usage_count>0 else 0.5
    
    def mutate(self)->str:
        """プロンプトを変異させる"""
        variations=[
            lambda t:t.replace("Explain","Describe in detail"),
            lambda t:t.replace("provide","give"),
            lambda t:t+"Consider multiple perspectives.",
            lambda t:f"Think step by step. {t}",
            lambda t:t.replace(".","with examples."),
        ]
        mutated=np.random.choice(variations)(self.template)
        self.mutations+=1
        return mutated

@dataclass
class Entity:
    name:str;type:str;count:int=1
    first_seen:datetime=field(default_factory=datetime.now)
    last_seen:datetime=field(default_factory=datetime.now)
    context:List[str]=field(default_factory=list)
    sentiment_history:List[float]=field(default_factory=list)

@dataclass
class LongTermMemory:
    entities:Dict[str,Entity]=field(default_factory=dict)
    facts:List[Dict]=field(default_factory=list)
    user_preferences:Dict[str,Any]=field(default_factory=dict)
    conversation_summaries:List[str]=field(default_factory=list)
    max_facts:int=200
    
    def add_entity(self,name:str,etype:str,context:str,sentiment:float=0):
        if name in self.entities:
            e=self.entities[name]
            e.count+=1;e.last_seen=datetime.now()
            e.sentiment_history.append(sentiment)
            if context not in e.context[-5:]:e.context.append(context)
            if len(e.context)>10:e.context=e.context[-10:]
        else:
            self.entities[name]=Entity(name=name,type=etype,context=[context],
                                      sentiment_history=[sentiment])
    
    def add_fact(self,fact:str,confidence:float=0.8,source:str="conversation"):
        self.facts.append({
            'fact':fact,'conf':confidence,'source':source,
            'ts':datetime.now().isoformat()
        })
        if len(self.facts)>self.max_facts:
            self.facts=self.facts[-self.max_facts:]
    
    def get_relevant_context(self,query:str,top_k:int=3)->str:
        words=set(re.findall(r'\b\w{3,}\b',query.lower()))
        relevant=[]
        for name,ent in self.entities.items():
            if any(w in name.lower()for w in words):
                avg_sent=statistics.mean(ent.sentiment_history)if ent.sentiment_history else 0
                relevant.append((f"{ent.name}({ent.type}):{ent.count}x",avg_sent))
        relevant.sort(key=lambda x:x[1],reverse=True)
        return" | ".join([r[0]for r in relevant[:top_k]])if relevant else""

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

@dataclass
class UserProfile:
    topics:Dict[str,int]=field(default_factory=lambda:defaultdict(int))
    avg_len:float=100.0;style:str="balanced";temp_pref:float=0.7
    pos_words:Set[str]=field(default_factory=set)
    neg_words:Set[str]=field(default_factory=set)
    feedback_hist:List[Dict]=field(default_factory=list)
    interaction_count:int=0
    last_updated:datetime=field(default_factory=datetime.now)
    intent_dist:Dict[str,int]=field(default_factory=lambda:defaultdict(int))
    time_of_day_pattern:Dict[int,int]=field(default_factory=lambda:defaultdict(int))
    complexity_preference:str="medium";learning_rate:float=0.1
    expertise_level:Dict[str,float]=field(default_factory=lambda:defaultdict(float))
    strategy_preference:Dict[str,float]=field(default_factory=lambda:defaultdict(float))
    reasoning_preference:Dict[str,int]=field(default_factory=lambda:defaultdict(int))
    
    def update_from_feedback(self,query:str,response:str,rating:int,
                           intent:Intent=None,complexity:Complexity=None,
                           strategy:Strategy=None):
        self.interaction_count+=1
        hour=datetime.now().hour
        self.time_of_day_pattern[hour]+=1
        
        self.feedback_hist.append({
            'q':query[:100],'r':response[:100],'rating':rating,
            'intent':intent,'complexity':complexity,'strategy':strategy,
            'ts':datetime.now().isoformat()
        })
        if len(self.feedback_hist)>200:
            self.feedback_hist=self.feedback_hist[-200:]
        
        words=re.findall(r'\b\w{4,}\b',query.lower())
        for w in words:
            self.topics[w]+=rating
            if rating>0:
                self.expertise_level[w]=min(1.0,self.expertise_level[w]+self.learning_rate)
            else:
                self.expertise_level[w]=max(0.0,self.expertise_level[w]-self.learning_rate*0.5)
        
        if intent:self.intent_dist[intent]+=1
        if strategy:self.strategy_preference[strategy]=self.strategy_preference.get(strategy,0.5)+rating*0.1
        
        resp_len=len(response)
        alpha=0.2 if rating>0 else 0.1
        self.avg_len=self.avg_len*(1-alpha)+resp_len*alpha
        
        if rating>0:
            if resp_len<150:self.style="concise"
            elif resp_len>500:self.style="detailed"
            else:self.style="balanced"
        
        resp_words=set(re.findall(r'\b\w{4,}\b',response.lower()))
        if rating>0:
            self.pos_words.update(w for w in resp_words if len(w)>4)
            if len(self.pos_words)>800:
                self.pos_words=set(list(self.pos_words)[-800:])
        elif rating<0:
            self.neg_words.update(w for w in resp_words if len(w)>4)
            if len(self.neg_words)>500:
                self.neg_words=set(list(self.neg_words)[-500:])
        
        self.last_updated=datetime.now()
    
    def get_adapted_temp(self)->float:
        if not self.feedback_hist:return 0.7
        recent=self.feedback_hist[-20:]
        avg_rating=sum(f.get('rating',0)for f in recent)/len(recent)if recent else 0
        expertise=sum(self.expertise_level.values())/len(self.expertise_level)if self.expertise_level else 0.5
        base_temp=0.7-expertise*0.2
        if avg_rating>0:return base_temp
        else:return min(1.0,base_temp+0.15)
    
    def get_style_prompt(self)->str:
        styles={
            'concise':'Be extremely concise. Use bullet points. Brief, actionable answers.',
            'balanced':'Provide clear, well-structured answers with appropriate detail.',
            'detailed':'Provide comprehensive explanations with examples, context, and deeper insights.'
        }
        return styles.get(self.style,styles['balanced'])
    
    def predict_intent(self,query:str)->Intent:
        q=query.lower()
        if any(w in q for w in['why','reason','because','cause']):return Intent.REASONING
        if any(w in q for w in['analyze','compare','evaluate']):return Intent.ANALYSIS
        if any(w in q for w in['?','how','what','when','where']):return Intent.QUESTION
        if any(w in q for w in['write','create','generate']):return Intent.CREATIVE
        if any(w in q for w in['code','program','algorithm']):return Intent.TECHNICAL
        if any(w in q for w in['explain','describe','detail']):return Intent.EXPLANATION
        return Intent.CASUAL
    
    def get_preferred_strategy(self)->Strategy:
        if not self.strategy_preference:return Strategy.DIRECT
        return Strategy(max(self.strategy_preference.items(),key=lambda x:x[1])[0])
    
    def get_expertise_topics(self,top_k:int=3)->List[Tuple[str,float]]:
        return sorted(self.expertise_level.items(),key=lambda x:x[1],reverse=True)[:top_k]

# ========== ロギング ==========
class Log:
    def __init__(self,n:str,l:str="INFO"):
        self.l=logging.getLogger(n);self.l.setLevel(getattr(logging,l))
        if not self.l.handlers:
            h=logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter('%(asctime)s|%(levelname)-8s|%(message)s','%H:%M:%S'))
            self.l.addHandler(h)

log=Log('llm')

# ========== ベクトルDB ==========
class VDB:
    def __init__(self,d=384):self.d=d;self.v=[]
    def _e(self,t):
        h=hashlib.sha256(t.encode()).digest();s=int.from_bytes(h[:4],'little')
        r=np.random.RandomState(s);v=r.randn(self.d).astype(np.float32);n=np.linalg.norm(v)
        return v/n if n>0 else v
    def add(self,i,t,m):
        e=self._e(t);m=m or{};m['text']=t;self.v.append((i,e,m))
    def get(self,q,k=1):
        if not self.v:return[]
        qe=self._e(q);rs=[(i,np.dot(qe,e),m)for i,e,m in self.v]
        rs.sort(key=lambda x:x[1],reverse=True);return rs[:k]

# ========== メインシステム ==========
class UltraAdvancedLLM:
    """Ultra-Advanced Self-Adapting LLM with Meta-Learning"""
    
    MODELS={
        'llama-3.1-8b-instant':{'speed':'fast','cost':'low','quality':'medium'},
        'llama-3.1-70b-versatile':{'speed':'medium','cost':'medium','quality':'high'},
        'llama-3.3-70b-versatile':{'speed':'medium','cost':'medium','quality':'high'},
    }
    
    def __init__(self,key:str=None,cfg:Cfg=None):
        self.key=key or os.environ.get('GROQ_API_KEY')
        if not self.key:raise ValueError("❌ GROQ_API_KEY required")
        self.cfg=cfg or Cfg();self.cli=Groq(api_key=self.key)
        self.db=VDB(self.cfg.dim)if self.cfg.vec_db else None
        
        # 基本統計
        self.n=0;self.ok=0;self.tok=0;self.cost=0;self.err=0
        self.t0=datetime.now()
        
        # コア機能
        self.profile=UserProfile()
        self.memory=LongTermMemory()if self.cfg.memory else None
        self.kg=KnowledgeGraph()if self.cfg.kg else None
        self.context_window:deque=deque(maxlen=15)
        
        # 高度な機能
        self.model_stats={m:ModelStats(name=m)for m in self.MODELS.keys()}
        self.total_pulls=0
        self.thompson=ThompsonSampler(list(self.MODELS.keys()))if self.cfg.thompson else None
        self.metacog=MetaCognition()if self.cfg.metacog else None
        
        # A/Bテスト
        self.ab_tests:Dict[str,ABTest]={}
        self.active_tests:List[str]=[]
        
        # プロンプト進化
        self.prompt_templates:List[PromptTemplate]=[]
        self._init_prompt_templates()
        
        # アンサンブル
        self.ensemble_history:List[Dict]=[]
        
        log.l.info(f"✅ Ultra-Advanced Init: {self.cfg.model}")
        features=[f"🧠Adapt:{self.cfg.adapt}",f"🎰MAB:{self.cfg.mab}",
                 f"💾Memory:{self.cfg.memory}",f"🧩KG:{self.cfg.kg}",
                 f"🤔CoT:{self.cfg.cot}",f"🔄Reflect:{self.cfg.reflection}",
                 f"🎲Thompson:{self.cfg.thompson}",f"📊AB:{self.cfg.ab_test}",
                 f"🎭Ensemble:{self.cfg.ensemble}",f"🧘MetaCog:{self.cfg.metacog}"]
        log.l.info(" | ".join(features))
    
    def _init_prompt_templates(self):
        """プロンプトテンプレート初期化"""
        templates=[
            PromptTemplate(id="cot_1",template="Let's think step by step.",category="reasoning"),
            PromptTemplate(id="cot_2",template="Break this down into smaller steps.",category="reasoning"),
            PromptTemplate(id="reflect_1",template="First, provide an answer. Then, reflect on potential issues.",category="reflection"),
            PromptTemplate(id="creative_1",template="Think creatively and outside the box.",category="creative"),
            PromptTemplate(id="technical_1",template="Provide technical details with code examples.",category="technical"),
        ]
        self.prompt_templates=templates
    
    def _select_strategy(self,intent:Intent,complexity:Complexity)->Strategy:
        """意図と複雑度から最適な戦略を選択"""
        if not self.cfg.adapt:return Strategy.DIRECT
        
        # 複雑度ベース
        if complexity==Complexity.EXPERT:
            if self.cfg.ensemble:return Strategy.ENSEMBLE
            elif self.cfg.cot:return Strategy.COT
        
        # 意図ベース
        if intent in[Intent.REASONING,Intent.ANALYSIS]:
            if self.cfg.cot:return Strategy.COT
        
        # ユーザー好みベース
        preferred=self.profile.get_preferred_strategy()
        if self.profile.strategy_preference.get(preferred,0)>0.7:
            return preferred
        
        return Strategy.DIRECT
    
    def _analyze_complexity(self,query:str)->Complexity:
        q=query.lower()
        tech_words=['algorithm','architecture','optimization','implementation']
        expert_words=['prove','derive','formal','theorem','axiom']
        multi_step=['step by step','first','then','finally','process']
        
        score=len(query)//100
        score+=sum(2 for w in tech_words if w in q)
        score+=sum(3 for w in expert_words if w in q)
        score+=sum(1 for w in multi_step if w in q)
        score+=q.count('?')
        
        if score<3:return Complexity.SIMPLE
        elif score<6:return Complexity.MEDIUM
        elif score<10:return Complexity.COMPLEX
        else:return Complexity.EXPERT
    
    def _analyze_sentiment(self,text:str)->float:
        pos_words=['good','great','excellent','thank','love','perfect','amazing','wonderful']
        neg_words=['bad','wrong','terrible','hate','awful','poor','fail','error']
        t=text.lower()
        pos=sum(t.count(w)for w in pos_words)
        neg=sum(t.count(w)for w in neg_words)
        total=pos+neg
        return(pos-neg)/total if total>0 else 0
    
    def _extract_entities_and_relations(self,text:str):
        """エンティティと関係を抽出"""
        if not self.memory or not self.kg:return
        
        # 簡易エンティティ抽出
        entities=re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',text)
        sentiment=self._analyze_sentiment(text)
        
        for ent in set(entities):
            if len(ent)>2:
                self.memory.add_entity(ent,'unknown',text[:100],sentiment)
                # 知識グラフに追加
                node_id=hashlib.md5(ent.encode()).hexdigest()[:8]
                if node_id not in self.kg.nodes:
                    self.kg.add_node(KnowledgeNode(
                        id=node_id,name=ent,type='entity',
                        properties={'mentions':1,'sentiment':sentiment}
                    ))
                else:
                    node=self.kg.nodes[node_id]
                    node.properties['mentions']=node.properties.get('mentions',0)+1
        
        # 簡易関係抽出(隣接エンティティ間)
        for i,ent1 in enumerate(entities):
            for ent2 in entities[i+1:i+3]:
                id1=hashlib.md5(ent1.encode()).hexdigest()[:8]
                id2=hashlib.md5(ent2.encode()).hexdigest()[:8]
                if id1 in self.kg.nodes and id2 in self.kg.nodes:
                    # 関係性を推測
                    relation="related_to"
                    self.kg.add_edge(KnowledgeEdge(
                        source=id1,target=id2,relation=relation,
                        weight=0.5+sentiment*0.2
                    ))
    
    def _select_model_thompson(self,complexity:Complexity)->str:
        """Thompson Samplingでモデル選択"""
        if not self.cfg.thompson or not self.thompson:
            return self._select_model_mab(complexity)
        
        context=f"{complexity.value}"
        model=self.thompson.select(context)
        log.l.info(f"🎲 Thompson: {model}")
        return model
    
    def _select_model_mab(self,complexity:Complexity)->str:
        """UCB1でモデル選択"""
        if not self.cfg.mab:return self.cfg.model
        
        exploration_rate=0.15
        if np.random.random()<exploration_rate:
            model=np.random.choice(list(self.MODELS.keys()))
            log.l.info(f"🎲 Explore: {model}")
            return model
        
        best_model=None;best_score=-float('inf')
        for name,stats in self.model_stats.items():
            score=stats.ucb1(self.total_pulls)
            if complexity==Complexity.EXPERT:
                if'70b'in name:score*=1.3
            elif complexity==Complexity.SIMPLE:
                if'8b'in name:score*=1.4
            
            if score>best_score:
                best_score=score;best_model=name
        
        log.l.info(f"🎯 Exploit: {best_model}(UCB:{best_score:.2f})")
        return best_model or self.cfg.model
    
    def _update_mab(self,model:str,reward:float,cost:float,latency:float,quality:float):
        """MAB統計を更新"""
        stats=self.model_stats[model]
        stats.pulls+=1;stats.total_reward+=reward
        stats.avg_cost=(stats.avg_cost*(stats.pulls-1)+cost)/stats.pulls
        stats.avg_latency=(stats.avg_latency*(stats.pulls-1)+latency)/stats.pulls
        stats.avg_quality=(stats.avg_quality*(stats.pulls-1)+quality)/stats.pulls
        if reward>0.6:stats.wins+=1
        stats.last_used=datetime.now()
        self.total_pulls+=1
        
        # Thompson Sampling更新
        if self.thompson:
            self.thompson.update(model,reward)
    
    def _cache(self,p)->Optional[Resp]:
        if not self.db:return None
        rs=self.db.get(p,1)
        if rs:
            i,s,m=rs[0]
            if s>=self.cfg.sim and time.time()-m.get('ts',0)<self.cfg.ttl:
                log.l.info(f"🔄 Cache:{s:.3f}");r=m.get('r',{})
                return Resp(text=r.get('text',''),conf=r.get('conf',0),
                           tok=r.get('tok',0),cost=r.get('cost',0),
                           model=r.get('model',''),lat=r.get('lat',0),
                           cache=True,sim=s)
        return None
    
    def _cost(self,m,p,c):
        pr={'llama-3.1-8b-instant':{'i':0.05/1e6,'o':0.08/1e6},
            'llama-3.1-70b-versatile':{'i':0.59/1e6,'o':0.79/1e6},
            'llama-3.3-70b-versatile':{'i':0.59/1e6,'o':0.79/1e6}}
        x=pr.get(m,{'i':0.0001/1e6,'o':0.0001/1e6});return p*x['i']+c*x['o']
    
    def _build_cot_prompt(self,query:str)->str:
        """Chain-of-Thought プロンプト構築"""
        templates=[t for t in self.prompt_templates if t.category=="reasoning"]
        if templates:
            # 最高性能のテンプレートを選択
            best=max(templates,key=lambda t:t.success_rate)
            best.usage_count+=1
            cot_instruction=best.template
        else:
            cot_instruction="Let's think step by step."
        
        return f"{cot_instruction}\n\nQuery: {query}\n\nReasoning:"
    
    def _build_reflection_prompt(self,query:str,initial_answer:str)->str:
        """Self-Reflection プロンプト"""
        return f"""Initial Answer: {initial_answer}

Now, reflect on this answer:
1. Are there any errors or inaccuracies?
2. What aspects could be improved?
3. Are there alternative perspectives?
4. What are the limitations?

Improved Answer:"""
    
    async def _execute_direct(self,query:str,model:str,temp:float,max_tok:int)->Resp:
        """直接実行戦略"""
        sys_prompt=self._build_advanced_prompt(query,self.profile.predict_intent(query),
                                                self._analyze_complexity(query))
        
        ar=await self._api_call(model,[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":query}
        ],temp,max_tok)
        
        return self._build_response(ar,model,Strategy.DIRECT)
    
    async def _execute_cot(self,query:str,model:str,temp:float,max_tok:int)->Resp:
        """Chain-of-Thought実行"""
        cot_prompt=self._build_cot_prompt(query)
        
        ar=await self._api_call(model,[
            {"role":"system","content":"You are a logical reasoning expert."},
            {"role":"user","content":cot_prompt}
        ],temp,max_tok)
        
        text=ar.choices[0].message.content or""
        
        # 推論ステップを抽出
        steps=re.findall(r'(?:Step \d+:|^\d+\.|^-)\s*(.+)',text,re.MULTILINE)
        
        resp=self._build_response(ar,model,Strategy.COT)
        resp.reasoning_steps=steps[:10]
        return resp
    
    async def _execute_reflection(self,query:str,model:str,temp:float,max_tok:int)->Resp:
        """Self-Reflection実行"""
        # まず初期回答を生成
        initial=await self._execute_direct(query,model,temp,max_tok)
        
        # 反省プロンプト
        reflect_prompt=self._build_reflection_prompt(query,initial.text)
        
        ar=await self._api_call(model,[
            {"role":"system","content":"You are a critical thinker who improves answers."},
            {"role":"user","content":reflect_prompt}
        ],temp,max_tok)
        
        resp=self._build_response(ar,model,Strategy.REFLECTION)
        resp.reflection=initial.text
        return resp
    
    async def _execute_ensemble(self,query:str,models:List[str],temp:float,max_tok:int)->Resp:
        """アンサンブル実行(複数モデル)"""
        responses=[]
        
        with ThreadPoolExecutor(max_workers=3)as executor:
            futures=[]
            for m in models[:3]:  # 最大3モデル
                future=executor.submit(lambda model:asyncio.run(
                    self._execute_direct(query,model,temp,max_tok)),m)
                futures.append((m,future))
            
            for m,future in futures:
                try:
                    r=future.result(timeout=30)
                    responses.append(r)
                except Exception as e:
                    log.l.warning(f"Ensemble fail:{m}:{e}")
        
        if not responses:
            # フォールバック
            return await self._execute_direct(query,self.cfg.model,temp,max_tok)
        
        # 最高信頼度の応答を選択
        best=max(responses,key=lambda r:r.conf)
        best.strategy=Strategy.ENSEMBLE
        best.alternatives=[r.text[:100]for r in responses if r!=best]
        
        # アンサンブル履歴
        self.ensemble_history.append({
            'query':query[:100],'models':[r.model for r in responses],
            'selected':best.model,'ts':datetime.now().isoformat()
        })
        
        return best
    
    async def _api_call(self,model:str,messages:List[Dict],temp:float,max_tok:int):
        """API呼び出しラッパー"""
        for attempt in range(self.cfg.retry):
            try:
                return self.cli.chat.completions.create(
                    model=model,messages=messages,
                    temperature=temp,max_tokens=max_tok
                )
            except(RateLimitError,APIError)as e:
                if attempt==self.cfg.retry-1:raise
                log.l.warning(f"Retry {attempt+1}/{self.cfg.retry}")
                await asyncio.sleep(self.cfg.delay*(2**attempt))
    
    def _build_response(self,ar,model:str,strategy:Strategy)->Resp:
        """APIレスポンスからRespオブジェクトを構築"""
        ch=ar.choices[0]
        txt=ch.message.content or""
        fr=ch.finish_reason
        co=self._cost(model,ar.usage.prompt_tokens,ar.usage.completion_tokens)
        
        sentiment=self._analyze_sentiment(txt)
        base_conf=0.90 if fr=="stop"else 0.75
        conf=base_conf*(1.0+sentiment*0.1)
        conf=max(0.0,min(1.0,conf))
        
        # 不確実性推定(簡易版)
        uncertainty=0.0
        uncertain_phrases=['maybe','perhaps','possibly','might','could be']
        uncertainty=sum(0.1 for p in uncertain_phrases if p in txt.lower())
        uncertainty=min(1.0,uncertainty)
        
        return Resp(
            text=txt,conf=conf,tok=ar.usage.total_tokens,
            pt=ar.usage.prompt_tokens,ct=ar.usage.completion_tokens,
            lat=0,cost=co,model=model,reason=fr,
            strategy=strategy,sentiment=sentiment,
            uncertainty=uncertainty
        )
    
    def _build_advanced_prompt(self,query:str,intent:Intent,complexity:Complexity)->str:
        """超高度な適応的システムプロンプト"""
        if not self.cfg.adapt:return "You are a helpful assistant."
        
        base="You are an advanced AI assistant with deep expertise."
        
        # スタイル
        style=self.profile.get_style_prompt()
        
        # 意図別調整
        intent_prompts={
            Intent.TECHNICAL:"Focus on technical accuracy. Provide code and algorithms.",
            Intent.CREATIVE:"Be creative and imaginative. Think unconventionally.",
            Intent.QUESTION:"Provide clear, direct answers with reasoning.",
            Intent.EXPLANATION:"Explain thoroughly with examples and analogies.",
            Intent.REASONING:"Use logical reasoning. Show your thinking process.",
            Intent.ANALYSIS:"Provide deep analysis. Compare and contrast.",
        }
        intent_adj=intent_prompts.get(intent,"")
        
        # 複雑度別
        complexity_prompts={
            Complexity.SIMPLE:"Keep it simple and clear.",
            Complexity.MEDIUM:"Provide adequate detail.",
            Complexity.COMPLEX:"Dive deep with comprehensive analysis.",
            Complexity.EXPERT:"Provide expert-level insights. Use formal terminology.",
        }
        complex_adj=complexity_prompts.get(complexity,"")
        
        # 専門度
        expertise=self.profile.get_expertise_topics(3)
        expert_ctx=""
        if expertise:
            topics=", ".join([t for t,s in expertise if s>0.6])
            if topics:expert_ctx=f" User expertise: {topics}."
        
        # 知識グラフコンテキスト
        kg_ctx=""
        if self.kg:
            # クエリからエンティティを抽出して関連概念を取得
            entities=re.findall(r'\b[A-Z][a-z]+\b',query)
            if entities:
                for ent in entities[:2]:
                    ent_id=hashlib.md5(ent.encode()).hexdigest()[:8]
                    if ent_id in self.kg.nodes:
                        related=self.kg.get_related_concepts(ent_id,3)
                        if related:
                            rel_names=[self.kg.nodes[r[0]].name for r in related]
                            kg_ctx+=f" {ent} relates to: {', '.join(rel_names)}."
        
        # 長期記憶
        memory_ctx=""
        if self.memory:
            rel=self.memory.get_relevant_context(query,3)
            if rel:memory_ctx=f" Context: {rel}."
        
        # 会話履歴
        history_ctx=""
        if self.context_window:
            recent=list(self.context_window)[-3:]
            if recent:history_ctx=f" Recent: {' | '.join(recent)}."
        
        # メタ認知
        metacog_ctx=""
        if self.metacog and self.metacog.should_seek_clarification():
            metacog_ctx=" If uncertain, ask clarifying questions."
        
        prompt=f"{base} {style} {intent_adj} {complex_adj}{expert_ctx}{kg_ctx}{memory_ctx}{history_ctx}{metacog_ctx}"
        
        # ネガティブワード回避
        if self.profile.neg_words:
            neg_sample=list(self.profile.neg_words)[:5]
            prompt+=f" Avoid: {', '.join(neg_sample)}."
        
        return prompt.strip()
    
    async def qa(self,p,**kw)->Resp:
        """メインクエリ処理"""
        self.n+=1;t0=time.time()
        
        try:
            if len(p)>self.cfg.max_len:
                return Resp(text=f"❌ 超過({self.cfg.max_len})",conf=0,reason="error")
            
            # キャッシュ
            c=self._cache(p)
            if c:return c
            
            # 分析
            intent=self.profile.predict_intent(p)
            complexity=self._analyze_complexity(p)
            strategy=self._select_strategy(intent,complexity)
            
            # モデル選択
            if self.cfg.thompson:
                m=self._select_model_thompson(complexity)
            else:
                m=self._select_model_mab(complexity)if self.cfg.mab else kw.get('model',self.cfg.model)
            
            # パラメータ
            temp=kw.get('temp',self.profile.get_adapted_temp()if self.cfg.adapt else self.cfg.temp)
            mt=kw.get('max_tok',self.cfg.max_tok)
            
            # 戦略実行
            if strategy==Strategy.COT and self.cfg.cot:
                r=await self._execute_cot(p,m,temp,mt)
            elif strategy==Strategy.REFLECTION and self.cfg.reflection:
                r=await self._execute_reflection(p,m,temp,mt)
            elif strategy==Strategy.ENSEMBLE and self.cfg.ensemble:
                models=sorted(self.model_stats.items(),key=lambda x:x[1].avg_reward,reverse=True)
                top_models=[name for name,_ in models[:3]]
                r=await self._execute_ensemble(p,top_models,temp,mt)
            else:
                r=await self._execute_direct(p,m,temp,mt)
            
            # メタデータ追加
            r.lat=(time.time()-t0)*1000
            r.intent=intent
            r.complexity=complexity
            
            if r.ok:self.ok+=1
            self.tok+=r.tok;self.cost+=r.cost
            
            # MAB更新
            quality=r.conf*(1.0-r.uncertainty*0.3)
            reward=(quality/max(r.cost,0.00001))*0.01
            self._update_mab(r.model,reward,r.cost,r.lat,quality)
            
            # コンテキスト
            self.context_window.append(p[:100])
            
            # エンティティ抽出
            self._extract_entities_and_relations(r.text)
            
            # メタ認知
            if self.metacog:
                self.metacog.add_result(r.conf,r.uncertainty,r.ok,"unknown")
            
            # キャッシュ保存
            if self.db and r.ok:
                did=hashlib.md5(p.encode()).hexdigest()
                self.db.add(did,p,{'r':r.dict(),'ts':time.time()})
            
            return r
            
        except Exception as e:
            self.err+=1;log.l.error(f"❌ {e}")
            return Resp(text=f"❌ {e}",conf=0,reason="error")
    
    def q(self,p,**kw)->Resp:return asyncio.run(self.qa(p,**kw))
    
    def add_feedback(self,query:str,response:str,rating:int,resp_obj:Resp=None):
        """高度なフィードバック処理"""
        if self.cfg.adapt:
            intent=resp_obj.intent if resp_obj else None
            complexity=resp_obj.complexity if resp_obj else None
            strategy=resp_obj.strategy if resp_obj else None
            self.profile.update_from_feedback(query,response,rating,intent,complexity,strategy)
            
            # プロンプトテンプレート更新
            if strategy==Strategy.COT:
                for t in self.prompt_templates:
                    if t.category=="reasoning"and t.usage_count>0:
                        if rating>0:t.success_count+=1
                        t.avg_quality=(t.avg_quality*(t.usage_count-1)+abs(rating))/t.usage_count
            
            # MABボーナス
            if resp_obj and self.cfg.mab:
                bonus=rating*0.15
                self.model_stats[resp_obj.model].total_reward+=bonus
            
            # Thompson Samplingボーナス
            if resp_obj and self.thompson:
                bonus_reward=0.7+rating*0.15
                self.thompson.update(resp_obj.model,bonus_reward)
            
            log.l.info(f"🧠 Deep Learning: {rating:+d} | {intent} | {complexity} | {strategy}")
    
    def create_ab_test(self,name:str,variant_a:Dict,variant_b:Dict):
        """A/Bテストを作成"""
        if not self.cfg.ab_test:return
        test=ABTest(name=name,variant_a=variant_a,variant_b=variant_b)
        self.ab_tests[name]=test
        self.active_tests.append(name)
        log.l.info(f"📊 A/B Test created: {name}")
    
    def add_ab_result(self,test_name:str,variant:str,value:float):
        """A/Bテスト結果を追加"""
        if test_name in self.ab_tests:
            self.ab_tests[test_name].add_result(variant,value)
    
    def get_ab_winner(self,test_name:str)->Optional[str]:
        """A/Bテスト勝者を判定"""
        if test_name in self.ab_tests:
            return self.ab_tests[test_name].get_winner()
        return None
    
    def stats(self)->Dict:
        """詳細統計"""
        up=(datetime.now()-self.t0).total_seconds()
        base_stats={
            'sys':{'up':up,'n':self.n,'ok':self.ok,'rate':self.ok/self.n if self.n>0 else 0},
            'api':{'n':self.n,'err':self.err,'rate':self.err/self.n if self.n>0 else 0,
                   'tok':self.tok,'cost':self.cost,'avg':self.cost/self.n if self.n>0 else 0,
                   'cache':len(self.db.v)if self.db else 0},
            'cfg':{'model':self.cfg.model,'adapt':self.cfg.adapt,'mab':self.cfg.mab,
                   'memory':self.cfg.memory,'kg':self.cfg.kg,'cot':self.cfg.cot,
                   'reflection':self.cfg.reflection,'ensemble':self.cfg.ensemble}
        }
        
        if self.cfg.adapt:
            base_stats['profile']={
                'style':self.profile.style,'avg_len':f"{self.profile.avg_len:.0f}",
                'temp':f"{self.profile.get_adapted_temp():.2f}",
                'interactions':self.profile.interaction_count,
                'expertise':len([e for e in self.profile.expertise_level.values()if e>0.5]),
                'preferred_strategy':self.profile.get_preferred_strategy()
            }
        
        if self.cfg.mab:
            mab_stats=[]
            for name,stats in sorted(self.model_stats.items(),
                                    key=lambda x:x[1].avg_reward,reverse=True):
                if stats.pulls>0:
                    mab_stats.append({
                        'model':name.split('-')[-1],'pulls':stats.pulls,
                        'win_rate':f"{stats.win_rate:.1%}",
                        'avg_reward':f"{stats.avg_reward:.4f}",
                        'avg_quality':f"{stats.avg_quality:.2f}",
                        'avg_cost':f"${stats.avg_cost:.6f}",
                        'avg_lat':f"{stats.avg_latency:.0f}ms"
                    })
            base_stats['mab']=mab_stats
        
        if self.cfg.memory and self.memory:
            top_ent=sorted(self.memory.entities.values(),key=lambda x:x.count,reverse=True)[:5]
            base_stats['memory']={
                'entities':len(self.memory.entities),
                'facts':len(self.memory.facts),
                'top_entities':[e.name for e in top_ent]
            }
        
        if self.cfg.kg and self.kg:
            base_stats['kg']={
                'nodes':len(self.kg.nodes),
                'edges':len(self.kg.edges),
                'avg_degree':len(self.kg.edges)*2/len(self.kg.nodes)if self.kg.nodes else 0
            }
        
        if self.cfg.metacog and self.metacog:
            base_stats['metacog']={
                'calibration':f"{self.metacog.get_calibration_score():.2f}",
                'should_clarify':self.metacog.should_seek_clarification(),
                'weaknesses':self.metacog.get_weakness_areas(3)
            }
        
        if self.cfg.ensemble and self.ensemble_history:
            base_stats['ensemble']={
                'total':len(self.ensemble_history),
                'last_models':self.ensemble_history[-1]['models']if self.ensemble_history else[]
            }
        
        if self.cfg.ab_test and self.ab_tests:
            ab_stats=[]
            for name,test in self.ab_tests.items():
                winner=test.get_winner()
                ab_stats.append({
                    'name':name,'a_n':len(test.results_a),'b_n':len(test.results_b),
                    'winner':winner or'undecided'
                })
            base_stats['ab_tests']=ab_stats
        
        return base_stats
    
    def save_all(self,base='ultra_llm'):
        """全データ保存"""
        # プロファイル
        try:
            data={
                'topics':dict(self.profile.topics),
                'avg_len':self.profile.avg_len,'style':self.profile.style,
                'temp_pref':self.profile.temp_pref,
                'pos_words':list(self.profile.pos_words),
                'neg_words':list(self.profile.neg_words),
                'feedback_hist':self.profile.feedback_hist,
                'interaction_count':self.profile.interaction_count,
                'intent_dist':dict(self.profile.intent_dist),
                'time_pattern':dict(self.profile.time_of_day_pattern),
                'expertise':dict(self.profile.expertise_level),
                'strategy_pref':dict(self.profile.strategy_preference),
                'last_updated':self.profile.last_updated.isoformat()
            }
            with open(f'{base}_profile.json','w',encoding='utf-8')as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            log.l.info(f"💾 Profile: {base}_profile.json")
        except Exception as e:
            log.l.error(f"❌ Profile save: {e}")
        
        # 長期記憶
        if self.memory:
            try:
                mem_data={
                    'entities':{k:{'name':v.name,'type':v.type,'count':v.count,
                                  'first':v.first_seen.isoformat(),
                                  'last':v.last_seen.isoformat(),
                                  'context':v.context,'sentiment':v.sentiment_history}
                               for k,v in self.memory.entities.items()},
                    'facts':self.memory.facts,
                    'preferences':self.memory.user_preferences
                }
                with open(f'{base}_memory.json','w',encoding='utf-8')as f:
                    json.dump(mem_data,f,ensure_ascii=False,indent=2)
                log.l.info(f"💾 Memory: {base}_memory.json")
            except Exception as e:
                log.l.error(f"❌ Memory save: {e}")
        
        # 知識グラフ
        if self.kg:
            try:
                kg_data={
                    'nodes':{k:{'name':v.name,'type':v.type,'properties':v.properties,
                               'confidence':v.confidence,'created':v.created.isoformat()}
                            for k,v in self.kg.nodes.items()},
                    'edges':[{'source':e.source,'target':e.target,'relation':e.relation,
                             'weight':e.weight,'properties':e.properties,
                             'created':e.created.isoformat()}
                            for e in self.kg.edges]
                }
                with open(f'{base}_kg.json','w',encoding='utf-8')as f:
                    json.dump(kg_data,f,ensure_ascii=False,indent=2)
                log.l.info(f"💾 KG: {base}_kg.json")
            except Exception as e:
                log.l.error(f"❌ KG save: {e}")
    
    def load_all(self,base='ultra_llm'):
        """全データ読み込み"""
        # プロファイル
        try:
            with open(f'{base}_profile.json','r',encoding='utf-8')as f:
                data=json.load(f)
            self.profile.topics=defaultdict(int,data.get('topics',{}))
            self.profile.avg_len=data.get('avg_len',100.0)
            self.profile.style=data.get('style','balanced')
            self.profile.temp_pref=data.get('temp_pref',0.7)
            self.profile.pos_words=set(data.get('pos_words',[]))
            self.profile.neg_words=set(data.get('neg_words',[]))
            self.profile.feedback_hist=data.get('feedback_hist',[])
            self.profile.interaction_count=data.get('interaction_count',0)
            self.profile.intent_dist=defaultdict(int,data.get('intent_dist',{}))
            self.profile.time_of_day_pattern=defaultdict(int,data.get('time_pattern',{}))
            self.profile.expertise_level=defaultdict(float,data.get('expertise',{}))
            self.profile.strategy_preference=defaultdict(float,data.get('strategy_pref',{}))
            log.l.info(f"📂 Profile loaded")
        except FileNotFoundError:
            log.l.info("ℹ️  Fresh start")
        except Exception as e:
            log.l.error(f"❌ Profile load: {e}")
        
        # 長期記憶
        if self.memory:
            try:
                with open(f'{base}_memory.json','r',encoding='utf-8')as f:
                    data=json.load(f)
                for k,v in data.get('entities',{}).items():
                    self.memory.entities[k]=Entity(
                        name=v['name'],type=v['type'],count=v['count'],
                        first_seen=datetime.fromisoformat(v['first']),
                        last_seen=datetime.fromisoformat(v['last']),
                        context=v['context'],
                        sentiment_history=v.get('sentiment',[])
                    )
                self.memory.facts=data.get('facts',[])
                self.memory.user_preferences=data.get('preferences',{})
                log.l.info(f"📂 Memory loaded")
            except FileNotFoundError:
                pass
            except Exception as e:
                log.l.error(f"❌ Memory load: {e}")
        
        # 知識グラフ
        if self.kg:
            try:
                with open(f'{base}_kg.json','r',encoding='utf-8')as f:
                    data=json.load(f)
                for k,v in data.get('nodes',{}).items():
                    self.kg.nodes[k]=KnowledgeNode(
                        id=k,name=v['name'],type=v['type'],
                        properties=v['properties'],
                        confidence=v['confidence'],
                        created=datetime.fromisoformat(v['created'])
                    )
                for e in data.get('edges',[]):
                    self.kg.edges.append(KnowledgeEdge(
                        source=e['source'],target=e['target'],
                        relation=e['relation'],weight=e['weight'],
                        properties=e['properties'],
                        created=datetime.fromisoformat(e['created'])
                    ))
                log.l.info(f"📂 KG loaded")
            except FileNotFoundError:
                pass
            except Exception as e:
                log.l.error(f"❌ KG load: {e}")

# ========== インタラクティブチャット ==========
class InteractiveChat:
    """高度なインタラクティブチャットインターフェース"""
    
    def __init__(self,llm:UltraAdvancedLLM):
        self.llm=llm
        self.history:List[Tuple[str,Resp]]=[]
        self.session_id=str(uuid.uuid4())[:8]
        
    def print_welcome(self):
        print("\n"+"="*70)
        print("🚀 Ultra-Advanced Self-Adapting LLM Chat System")
        print("="*70)
        print("コマンド:")
        print("  /stats    - 統計情報表示")
        print("  /save     - データ保存")
        print("  /load     - データ読み込み")
        print("  /feedback <rating> - 最後の回答に評価 (-2 to +2)")
        print("  /clear    - 履歴クリア")
        print("  /help     - ヘルプ表示")
        print("  /exit     - 終了")
        print("="*70+"\n")
    
    def print_response(self,resp:Resp,query:str):
        print(f"\n🤖 Assistant [{resp.model.split('-')[-1]}]:")
        print("-"*70)
        print(resp.text)
        print("-"*70)
        
        # メタデータ
        meta=[]
        if resp.strategy:meta.append(f"📋{resp.strategy.value}")
        if resp.intent:meta.append(f"🎯{resp.intent.value}")
        if resp.complexity:meta.append(f"⚙️{resp.complexity.value}")
        meta.append(f"✅{resp.conf:.2f}")
        meta.append(f"🎲{resp.uncertainty:.2f}")
        meta.append(f"💰${resp.cost:.6f}")
        meta.append(f"⏱️{resp.lat:.0f}ms")
        meta.append(f"🎫{resp.tok}tok")
        if resp.cache:meta.append(f"🔄Cache({resp.sim:.2f})")
        
        print(" | ".join(meta))
        
        # 推論ステップ
        if resp.reasoning_steps:
            print(f"\n🧠 Reasoning Steps:")
            for i,step in enumerate(resp.reasoning_steps[:3],1):
                print(f"  {i}. {step[:80]}...")
        
        # 反省
        if resp.reflection:
            print(f"\n🔄 Initial thought: {resp.reflection[:100]}...")
        
        # 代替案
        if resp.alternatives:
            print(f"\n🎭 Alternatives considered: {len(resp.alternatives)}")
        
        print()
    
    def print_stats(self):
        stats=self.llm.stats()
        print("\n"+"="*70)
        print("📊 System Statistics")
        print("="*70)
        
        # システム統計
        sys=stats['sys']
        print(f"\n⏱️  Uptime: {sys['up']:.1f}s | Queries: {sys['n']} | Success: {sys['rate']:.1%}")
        
        # API統計
        api=stats['api']
        print(f"🌐 API: {api['n']} calls | Errors: {api['err']} ({api['rate']:.1%})")
        print(f"💰 Cost: ${api['cost']:.6f} (avg: ${api['avg']:.6f}) | Cache: {api['cache']}")
        
        # プロフィール
        if 'profile' in stats:
            prof=stats['profile']
            print(f"\n👤 Profile: style={prof['style']} | len={prof['avg_len']} | temp={prof['temp']}")
            print(f"   Interactions: {prof['interactions']} | Expertise areas: {prof['expertise']}")
            print(f"   Preferred strategy: {prof['preferred_strategy']}")
        
        # MAB統計
        if 'mab' in stats and stats['mab']:
            print(f"\n🎰 Multi-Armed Bandit:")
            for m in stats['mab']:
                print(f"   {m['model']:12s}: pulls={m['pulls']:3d} win={m['win_rate']} "
                      f"reward={m['avg_reward']} quality={m['avg_quality']} "
                      f"cost={m['avg_cost']} lat={m['avg_lat']}")
        
        # メモリ
        if 'memory' in stats:
            mem=stats['memory']
            print(f"\n💾 Memory: {mem['entities']} entities | {mem['facts']} facts")
            if mem['top_entities']:
                print(f"   Top: {', '.join(mem['top_entities'][:3])}")
        
        # 知識グラフ
        if 'kg' in stats:
            kg=stats['kg']
            print(f"\n🧩 Knowledge Graph: {kg['nodes']} nodes | {kg['edges']} edges | "
                  f"avg_degree={kg['avg_degree']:.2f}")
        
        # メタ認知
        if 'metacog' in stats:
            mc=stats['metacog']
            print(f"\n🧘 Meta-Cognition: calibration={mc['calibration']} | "
                  f"clarify={mc['should_clarify']}")
            if mc['weaknesses']:
                print(f"   Weaknesses: {', '.join(mc['weaknesses'])}")
        
        # アンサンブル
        if 'ensemble' in stats:
            ens=stats['ensemble']
            print(f"\n🎭 Ensemble: {ens['total']} runs")
            if ens['last_models']:
                print(f"   Last models: {', '.join(ens['last_models'])}")
        
        # A/Bテスト
        if 'ab_tests' in stats:
            print(f"\n📊 A/B Tests:")
            for t in stats['ab_tests']:
                print(f"   {t['name']}: A={t['a_n']} B={t['b_n']} winner={t['winner']}")
        
        print("="*70+"\n")
    
    def handle_command(self,cmd:str)->bool:
        """コマンド処理。継続する場合True、終了する場合False"""
        parts=cmd.strip().split()
        command=parts[0].lower()
        
        if command=='/exit':
            print("👋 Goodbye!")
            return False
        
        elif command=='/stats':
            self.print_stats()
        
        elif command=='/save':
            base=parts[1]if len(parts)>1 else'ultra_llm'
            self.llm.save_all(base)
            print(f"💾 Saved to {base}_*.json")
        
        elif command=='/load':
            base=parts[1]if len(parts)>1 else'ultra_llm'
            self.llm.load_all(base)
            print(f"📂 Loaded from {base}_*.json")
        
        elif command=='/feedback':
            if len(self.history)==0:
                print("❌ No previous response to rate")
                return True
            
            try:
                rating=int(parts[1])if len(parts)>1 else 0
                if rating<-2 or rating>2:
                    print("❌ Rating must be between -2 and +2")
                    return True
                
                last_q,last_r=self.history[-1]
                self.llm.add_feedback(last_q,last_r.text,rating,last_r)
                print(f"✅ Feedback recorded: {rating:+d}")
                
            except ValueError:
                print("❌ Invalid rating format")
        
        elif command=='/clear':
            self.history.clear()
            self.llm.context_window.clear()
            print("🗑️  History cleared")
        
        elif command=='/help':
            self.print_welcome()
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands")
        
        return True
    
    def run(self):
        """メインループ"""
        self.print_welcome()
        
        while True:
            try:
                query=input("👤 You: ").strip()
                
                if not query:
                    continue
                
                # コマンド処理
                if query.startswith('/'):
                    if not self.handle_command(query):
                        break
                    continue
                
                # クエリ処理
                print("\n⏳ Thinking...")
                resp=self.llm.q(query)
                
                self.history.append((query,resp))
                self.print_response(resp,query)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type /exit to quit or continue chatting.")
                continue
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                log.l.error(f"Chat error: {e}")

# ========== メイン実行 ==========
def main():
    """エントリーポイント"""
    import argparse
    
    parser=argparse.ArgumentParser(description='Ultra-Advanced Self-Adapting LLM')
    parser.add_argument('--model',default='llama-3.1-8b-instant',
                       help='Base model to use')
    parser.add_argument('--no-adapt',action='store_true',
                       help='Disable adaptation')
    parser.add_argument('--no-mab',action='store_true',
                       help='Disable multi-armed bandit')
    parser.add_argument('--no-memory',action='store_true',
                       help='Disable long-term memory')
    parser.add_argument('--no-kg',action='store_true',
                       help='Disable knowledge graph')
    parser.add_argument('--no-cot',action='store_true',
                       help='Disable chain-of-thought')
    parser.add_argument('--no-reflection',action='store_true',
                       help='Disable self-reflection')
    parser.add_argument('--no-ensemble',action='store_true',
                       help='Disable ensemble learning')
    parser.add_argument('--no-thompson',action='store_true',
                       help='Disable Thompson sampling')
    parser.add_argument('--query',type=str,help='Single query mode')
    parser.add_argument('--load',type=str,help='Load saved data')
    
    args=parser.parse_args()
    
    # 設定
    cfg=Cfg(
        model=args.model,
        adapt=not args.no_adapt,
        mab=not args.no_mab,
        memory=not args.no_memory,
        kg=not args.no_kg,
        cot=not args.no_cot,
        reflection=not args.no_reflection,
        ensemble=not args.no_ensemble,
        thompson=not args.no_thompson
    )
    
    # システム初期化
    try:
        llm=UltraAdvancedLLM(cfg=cfg)
        
        # データ読み込み
        if args.load:
            llm.load_all(args.load)
        
        # シングルクエリモード
        if args.query:
            resp=llm.q(args.query)
            print(resp.text)
            print(f"\nMetadata: conf={resp.conf:.2f} cost=${resp.cost:.6f} "
                  f"lat={resp.lat:.0f}ms tok={resp.tok}")
            return
        
        # インタラクティブモード
        chat=InteractiveChat(llm)
        chat.run()
        
        # 終了時に保存
        print("\n💾 Saving session data...")
        llm.save_all()
        
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("Please set GROQ_API_KEY environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        log.l.error(f"Fatal: {e}")
        sys.exit(1)

if __name__=='__main__':
    main()
