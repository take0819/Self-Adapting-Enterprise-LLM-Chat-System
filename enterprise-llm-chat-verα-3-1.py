# Constitutional AI check
            if self.constitutional:
                const_scores=self.constitutional.evaluate_response(r.text,p)
                compliance=self.constitutional.get_compliance_score(const_scores)
                
                if compliance<0.6:  # 基準を下回る場合は修正
                    improve_prompt=f"Revise this response to better align with ethical principles:\n{r.text}"
                    improved_ar=await self._api_call(m,[
                        {"role":"system","content":"Revise to be more helpful, harmless, and honest."},
                        {"role":"user","content":improve_prompt}
                    ],temp,mt)
                    r=self._build_response(improved_ar,m,r.strategy or Strategy.DIRECT)
                
                r.sentiment=compliance  # コンプライアンスをsentimentに格納
            
            # Causal reasoning
            if self.causal and any(w in p.lower()for w in['why','cause','because','reason']):
                # 因果関係を抽出
                causes=re.findall(r'because\s+(.+?)(?:[,.]|$)',r.text.lower())
                if causes:
                    for cause in causes[:3]:
                        link=CausalLink(
                            cause=cause.strip(),
                            effect=p[:50],
                            strength=0.7,
                            evidence=[r.text[:100]]
                        )
                        self.causal.add_causal_link(link)
            
            # Analogy engine
            if self.analogy and intent==Intent.EXPLANATION:
                # ドメイン知識を蓄積
                domain=intent.value
                knowledge_points=re.findall(r'[A-Z][^.!?]*[.!?]',r.text)
                if knowledge_points:
                    self.analogy.add_domain_knowledge(domain,knowledge_points[:5])
                
                # アナロジーを探索
                analogy=self.analogy.find_analogy(domain,p)
                if analogy and analogy.similarity>0.5:
                    transferred=self.analogy.transfer_knowledge(analogy)
                    if transferred:
                        r.alternatives.extend(transferred[:2])
            
            # World model update
            if self.world_model:
                # 状態更新
                effects={'last_query':p[:50],'last_response':r.text[:50]}
                self.world_model.update_state(f"answered:{intent.value if intent else'query'}",effects)
                
                # 信念更新
                if r.conf>0.8:
                    self.world_model.current_state.update_belief(p[:100],r.conf)
            
            # Self-play learning
            if self.self_play and complexity==Complexity.EXPERT:
                # 別の戦略で回答を生成
                game=self.self_play.create_game(p)
                game.agent_a_answer=r.text
                
                # Agent B: より保守的な回答
                alt_ar=await self._api_call(m,[
                    {"role":"system","content":"Be conservative and acknowledge limitations."},
                    {"role":"user","content":p}
                ],temp*0.5,mt)
                game.agent_b_answer=alt_ar.choices[0].message.content or""
                
                self.self_play.play_game(game,r.text,game.agent_b_answer)
                
                # 簡易判定
                winner=self.self_play.judge_winner(game,"Comprehensive wins")
                if winner=="B":
                    # Agent Bが勝った場合はその回答を採用
                    r.text=game.agent_b_answer
                    r.alternatives=[game.agent_a_answer[:100]]
            
            # Meta-learning
            if self.meta_learner:
                task=f"{intent.value}_{complexity.value}"if intent else"general"
                performance=r.conf*(1.0-r.uncertainty*0.3)
                self.meta_learner.record_learning(task,performance)
                
                # 戦略提案
                suggested_strategy=self.meta_learner.suggest_strategy(task,performance)
                if suggested_strategy!=strategy.value:
                    log.l.info(f"🧬 Meta-learner suggests: {suggested_strategy} (current: {strategy.value})")
            
            # 信頼度キャリブレーション
            if self.calibrator:
                r.conf=self.calibrator.calibrate(r.conf)#!/usr/bin/env python3
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

class MemoryType(str,Enum):
    EPISODIC="episodic";SEMANTIC="semantic";PROCEDURAL="procedural";WORKING="working"

class LearningMode(str,Enum):
    EXPLORATION="exploration";EXPLOITATION="exploitation";BALANCED="balanced"

class ReasoningPath(str,Enum):
    FORWARD="forward";BACKWARD="backward";BIDIRECTIONAL="bidirectional"

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
    # Ultra-advanced features
    multi_hop:bool=True;debate:bool=True;critic:bool=True
    tree_of_thoughts:bool=True;rag:bool=True;confidence_calibration:bool=True
    active_learning:bool=True;curriculum:bool=True
    # Next-gen features
    self_play:bool=True;constitutional_ai:bool=True;few_shot_learning:bool=True
    meta_learning:bool=True;neuro_symbolic:bool=True;causal_reasoning:bool=True
    world_model:bool=True;counterfactual:bool=True;analogy_engine:bool=True

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
class ThoughtNode:
    """Tree of Thoughts用ノード"""
    id:str;content:str;parent:Optional[str]=None
    children:List[str]=field(default_factory=list)
    value:float=0.0;visits:int=0;depth:int=0
    
    def ucb_score(self,parent_visits:int,c:float=1.414)->float:
        if self.visits==0:return float('inf')
        exploitation=self.value/self.visits
        exploration=c*math.sqrt(math.log(parent_visits)/self.visits)
        return exploitation+exploration

@dataclass
class DebateArgument:
    """討論システムの引数"""
    position:str;argument:str;confidence:float
    evidence:List[str]=field(default_factory=list)
    counterarguments:List[str]=field(default_factory=list)
    timestamp:datetime=field(default_factory=datetime.now)

@dataclass
class CriticFeedback:
    """批評システムのフィードバック"""
    aspect:str;score:float;comments:str
    suggestions:List[str]=field(default_factory=list)
    severity:str="medium"  # low, medium, high, critical

@dataclass
class ActiveLearningQuery:
    """能動学習のクエリ"""
    query:str;uncertainty:float;expected_gain:float
    priority:float;asked:bool=False
    response:Optional[str]=None

@dataclass
class ConstitutionalRule:
    """憲法的AIのルール"""
    id:str;principle:str;weight:float=1.0
    violations:int=0;applied:int=0
    category:str="general"

@dataclass
class WorldState:
    """世界モデルの状態"""
    entities:Dict[str,Any]=field(default_factory=dict)
    relations:List[Tuple[str,str,str]]=field(default_factory=list)
    beliefs:Dict[str,float]=field(default_factory=dict)
    timestamp:datetime=field(default_factory=datetime.now)
    
    def update_belief(self,statement:str,confidence:float):
        self.beliefs[statement]=confidence
    
    def get_belief(self,statement:str)->float:
        return self.beliefs.get(statement,0.5)

@dataclass
class CausalLink:
    """因果関係リンク"""
    cause:str;effect:str;strength:float
    evidence:List[str]=field(default_factory=list)
    confounders:List[str]=field(default_factory=list)
    mechanism:Optional[str]=None

@dataclass
class Analogy:
    """アナロジー(類推)"""
    source_domain:str;target_domain:str
    mappings:Dict[str,str]=field(default_factory=dict)
    similarity:float=0.0
    transferable_knowledge:List[str]=field(default_factory=list)

@dataclass
class FewShotExample:
    """Few-shot学習の例"""
    input:str;output:str;quality:float=1.0
    context:Optional[str]=None
    timestamp:datetime=field(default_factory=datetime.now)

@dataclass
class SelfPlayGame:
    """自己対戦ゲーム"""
    id:str;question:str
    agent_a_answer:str;agent_b_answer:str
    winner:Optional[str]=None
    judge_reasoning:Optional[str]=None
    improvement_gained:float=0.0

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
    def get_by_threshold(self,q,threshold:float=0.7)->List:
        """閾値以上の類似度を持つ全ドキュメントを取得"""
        if not self.v:return[]
        qe=self._e(q);rs=[(i,np.dot(qe,e),m)for i,e,m in self.v if np.dot(qe,e)>=threshold]
        rs.sort(key=lambda x:x[1],reverse=True);return rs

# ========== 高度なシステムコンポーネント ==========
class TreeOfThoughts:
    """Tree of Thoughts推論システム"""
    def __init__(self,max_depth:int=3,branching:int=3):
        self.nodes:Dict[str,ThoughtNode]={}
        self.max_depth=max_depth
        self.branching=branching
        self.root_id:Optional[str]=None
    
    def create_root(self,content:str)->str:
        node_id=str(uuid.uuid4())[:8]
        self.nodes[node_id]=ThoughtNode(id=node_id,content=content,depth=0)
        self.root_id=node_id
        return node_id
    
    def expand(self,node_id:str,children_content:List[str]):
        """ノードを展開"""
        if node_id not in self.nodes:return
        parent=self.nodes[node_id]
        if parent.depth>=self.max_depth:return
        
        for content in children_content[:self.branching]:
            child_id=str(uuid.uuid4())[:8]
            child=ThoughtNode(
                id=child_id,content=content,parent=node_id,
                depth=parent.depth+1
            )
            self.nodes[child_id]=child
            parent.children.append(child_id)
    
    def backpropagate(self,node_id:str,value:float):
        """価値を逆伝播"""
        current_id=node_id
        while current_id:
            node=self.nodes[current_id]
            node.visits+=1
            node.value+=value
            current_id=node.parent
    
    def select_best_path(self)->List[str]:
        """最良のパスを選択"""
        if not self.root_id:return[]
        path=[self.root_id]
        current_id=self.root_id
        
        while True:
            node=self.nodes[current_id]
            if not node.children:break
            
            # 最も価値の高い子を選択
            best_child_id=max(
                node.children,
                key=lambda cid:self.nodes[cid].value/max(self.nodes[cid].visits,1)
            )
            path.append(best_child_id)
            current_id=best_child_id
        
        return path
    
    def get_path_content(self,path:List[str])->List[str]:
        return[self.nodes[nid].content for nid in path]

class DebateSystem:
    """多視点討論システム"""
    def __init__(self):
        self.arguments:List[DebateArgument]=[]
        self.positions:Dict[str,List[DebateArgument]]=defaultdict(list)
    
    def add_argument(self,arg:DebateArgument):
        self.arguments.append(arg)
        self.positions[arg.position].append(arg)
    
    def get_strongest_position(self)->Optional[str]:
        if not self.positions:return None
        scores={}
        for pos,args in self.positions.items():
            scores[pos]=sum(a.confidence for a in args)/len(args)
        return max(scores.items(),key=lambda x:x[1])[0]
    
    def synthesize(self)->str:
        """全視点を統合"""
        if not self.arguments:return""
        synthesis=[]
        for pos,args in self.positions.items():
            best_arg=max(args,key=lambda a:a.confidence)
            synthesis.append(f"{pos}: {best_arg.argument}")
        return"\n".join(synthesis)

class CriticSystem:
    """批判的評価システム"""
    def __init__(self):
        self.feedbacks:List[CriticFeedback]=[]
        self.aspects=['accuracy','completeness','clarity','relevance','depth']
    
    def evaluate(self,text:str,query:str)->Dict[str,float]:
        """テキストを多角的に評価"""
        scores={}
        # 簡易評価ロジック
        words=len(text.split())
        scores['completeness']=min(1.0,words/200)
        scores['clarity']=0.8 if'.'in text else 0.5
        scores['relevance']=0.7  # 実際はクエリとの類似度を計算
        scores['accuracy']=0.8  # 実際はファクトチェックが必要
        scores['depth']=min(1.0,text.count('\n')/5)
        return scores
    
    def add_feedback(self,feedback:CriticFeedback):
        self.feedbacks.append(feedback)
    
    def get_overall_score(self)->float:
        if not self.feedbacks:return 0.5
        return statistics.mean(f.score for f in self.feedbacks)

class ConfidenceCalibrator:
    """信頼度キャリブレーションシステム"""
    def __init__(self):
        self.predictions:List[Tuple[float,bool]]=[]  # (confidence, correct)
        self.bins:int=10
    
    def add_prediction(self,confidence:float,correct:bool):
        self.predictions.append((confidence,correct))
    
    def get_calibration_error(self)->float:
        """Expected Calibration Error (ECE)を計算"""
        if len(self.predictions)<10:return 0.0
        
        bin_edges=np.linspace(0,1,self.bins+1)
        ece=0.0
        
        for i in range(self.bins):
            bin_preds=[(c,cor)for c,cor in self.predictions 
                      if bin_edges[i]<=c<bin_edges[i+1]]
            if not bin_preds:continue
            
            avg_conf=statistics.mean(c for c,_ in bin_preds)
            accuracy=sum(1 for _,cor in bin_preds if cor)/len(bin_preds)
            ece+=abs(avg_conf-accuracy)*len(bin_preds)/len(self.predictions)
        
        return ece
    
    def calibrate(self,raw_confidence:float)->float:
        """信頼度を補正"""
        if len(self.predictions)<20:return raw_confidence
        
        # 簡易的なプラットスケーリング
        ece=self.get_calibration_error()
        if ece>0.1:
            return raw_confidence*0.9  # 過信を抑制
        return raw_confidence

class CurriculumLearner:
    """カリキュラム学習システム"""
    def __init__(self):
        self.difficulty_levels:Dict[str,float]=defaultdict(lambda:0.5)
        self.mastery:Dict[str,float]=defaultdict(lambda:0.0)
        self.learning_history:List[Dict]=[]
    
    def update_mastery(self,topic:str,success:bool,difficulty:float):
        """習熟度を更新"""
        delta=0.1 if success else -0.05
        self.mastery[topic]=max(0.0,min(1.0,self.mastery[topic]+delta))
        self.difficulty_levels[topic]=difficulty
        
        self.learning_history.append({
            'topic':topic,'success':success,'difficulty':difficulty,
            'mastery':self.mastery[topic],'ts':datetime.now().isoformat()
        })
    
    def get_next_difficulty(self,topic:str)->float:
        """次の最適難易度を推奨"""
        current_mastery=self.mastery[topic]
        # Zone of Proximal Development: 現在のレベルより少し上
        return min(1.0,current_mastery+0.2)
    
    def should_review(self,topic:str)->bool:
        """復習が必要かを判定"""
        if topic not in self.learning_history:return False
        recent=[h for h in self.learning_history[-10:]if h['topic']==topic]
        if not recent:return False
        recent_success=sum(1 for h in recent if h['success'])/len(recent)
        return recent_success<0.7

# ========== Next-Gen AI Components ==========
class ConstitutionalAI:
    """憲法的AI - 価値観に基づく意思決定"""
    def __init__(self):
        self.rules:List[ConstitutionalRule]=[]
        self._init_default_rules()
    
    def _init_default_rules(self):
        """デフォルトの憲法的ルールを初期化"""
        default_rules=[
            ConstitutionalRule(id="helpful",principle="Be helpful and informative",weight=1.0,category="core"),
            ConstitutionalRule(id="harmless",principle="Avoid harmful or dangerous advice",weight=1.5,category="safety"),
            ConstitutionalRule(id="honest",principle="Be truthful and acknowledge uncertainty",weight=1.2,category="core"),
            ConstitutionalRule(id="fair",principle="Treat all people fairly and without bias",weight=1.3,category="ethics"),
            ConstitutionalRule(id="privacy",principle="Respect privacy and confidentiality",weight=1.4,category="ethics"),
        ]
        self.rules.extend(default_rules)
    
    def evaluate_response(self,response:str,context:str)->Dict[str,float]:
        """応答を憲法的ルールに照らして評価"""
        scores={}
        for rule in self.rules:
            # 簡易評価ロジック(実際はLLMで評価)
            score=0.8  # デフォルトスコア
            
            if rule.id=="harmless":
                dangerous_words=['kill','harm','illegal','hack','exploit']
                if any(w in response.lower()for w in dangerous_words):
                    score=0.3
            
            elif rule.id=="honest":
                uncertain_phrases=['I think','maybe','possibly','not sure']
                has_uncertainty=any(p in response.lower()for p in uncertain_phrases)
                score=0.9 if has_uncertainty else 0.7
            
            scores[rule.id]=score*rule.weight
            rule.applied+=1
            if score<0.5:
                rule.violations+=1
        
        return scores
    
    def get_compliance_score(self,scores:Dict[str,float])->float:
        """全体のコンプライアンススコアを計算"""
        if not scores:return 0.5
        return statistics.mean(scores.values())/max(r.weight for r in self.rules)

class WorldModel:
    """世界モデル - 状態予測とシミュレーション"""
    def __init__(self):
        self.states:List[WorldState]=[]
        self.current_state:WorldState=WorldState()
        self.transitions:List[Tuple[WorldState,str,WorldState]]=[]
    
    def update_state(self,action:str,effects:Dict[str,Any]):
        """アクションによる状態更新"""
        new_state=WorldState(
            entities=self.current_state.entities.copy(),
            relations=self.current_state.relations.copy(),
            beliefs=self.current_state.beliefs.copy()
        )
        
        for key,value in effects.items():
            new_state.entities[key]=value
        
        self.transitions.append((self.current_state,action,new_state))
        self.states.append(new_state)
        self.current_state=new_state
    
    def predict_outcome(self,action:str)->Dict[str,float]:
        """アクションの結果を予測"""
        # 過去の類似アクションから予測
        similar_transitions=[t for t in self.transitions if action in t[1]]
        
        if not similar_transitions:
            return{'success':0.5,'impact':0.5}
        
        # 簡易的な予測
        return{
            'success':0.7,
            'impact':0.6,
            'confidence':len(similar_transitions)/10
        }
    
    def simulate_counterfactual(self,what_if:str)->WorldState:
        """反事実的シミュレーション"""
        counterfactual_state=WorldState(
            entities=self.current_state.entities.copy(),
            relations=self.current_state.relations.copy(),
            beliefs=self.current_state.beliefs.copy()
        )
        # 反事実的な変更を適用
        counterfactual_state.update_belief(what_if,0.8)
        return counterfactual_state

class CausalReasoning:
    """因果推論システム"""
    def __init__(self):
        self.causal_links:List[CausalLink]=[]
        self.causal_graph:Dict[str,List[str]]=defaultdict(list)
    
    def add_causal_link(self,link:CausalLink):
        """因果リンクを追加"""
        self.causal_links.append(link)
        self.causal_graph[link.cause].append(link.effect)
    
    def find_causes(self,effect:str,max_depth:int=3)->List[List[str]]:
        """効果の原因を探索"""
        paths=[]
        
        def dfs(current:str,path:List[str],depth:int):
            if depth>max_depth:return
            if current==effect and len(path)>1:
                paths.append(path.copy())
                return
            
            for link in self.causal_links:
                if link.effect==current and link.cause not in path:
                    path.append(link.cause)
                    dfs(link.cause,path,depth+1)
                    path.pop()
        
        for link in self.causal_links:
            if link.effect==effect:
                dfs(link.cause,[effect,link.cause],1)
        
        return paths
    
    def estimate_causal_strength(self,cause:str,effect:str)->float:
        """因果強度を推定"""
        links=[l for l in self.causal_links if l.cause==cause and l.effect==effect]
        if not links:return 0.0
        return max(l.strength for l in links)
    
    def find_confounders(self,cause:str,effect:str)->List[str]:
        """交絡因子を発見"""
        confounders=set()
        for link in self.causal_links:
            if link.cause==cause and link.effect==effect:
                confounders.update(link.confounders)
        return list(confounders)

class AnalogyEngine:
    """アナロジー推論エンジン"""
    def __init__(self):
        self.analogies:List[Analogy]=[]
        self.domain_knowledge:Dict[str,List[str]]=defaultdict(list)
    
    def find_analogy(self,target_domain:str,target_problem:str)->Optional[Analogy]:
        """類似の問題領域を探索"""
        # ドメイン間の類似度を計算
        best_analogy=None
        best_similarity=0.0
        
        for source_domain,features in self.domain_knowledge.items():
            if source_domain==target_domain:continue
            
            # 簡易類似度計算
            target_words=set(target_problem.lower().split())
            source_words=set(' '.join(features).lower().split())
            similarity=len(target_words&source_words)/max(len(target_words|source_words),1)
            
            if similarity>best_similarity:
                best_similarity=similarity
                best_analogy=Analogy(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    similarity=similarity
                )
        
        return best_analogy if best_similarity>0.3 else None
    
    def transfer_knowledge(self,analogy:Analogy)->List[str]:
        """知識転移"""
        source_knowledge=self.domain_knowledge.get(analogy.source_domain,[])
        return source_knowledge[:5]  # 上位5つの知識を転移
    
    def add_domain_knowledge(self,domain:str,knowledge:List[str]):
        """ドメイン知識を追加"""
        self.domain_knowledge[domain].extend(knowledge)

class FewShotLearner:
    """Few-shot学習システム"""
    def __init__(self,max_examples:int=10):
        self.examples:List[FewShotExample]=[]
        self.max_examples=max_examples
        self.performance_history:List[float]=[]
    
    def add_example(self,example:FewShotExample):
        """例を追加"""
        self.examples.append(example)
        
        # 品質でソートして上位を保持
        if len(self.examples)>self.max_examples:
            self.examples=sorted(self.examples,key=lambda x:x.quality,reverse=True)[:self.max_examples]
    
    def get_relevant_examples(self,query:str,k:int=3)->List[FewShotExample]:
        """関連する例を取得"""
        # 簡易的な類似度計算
        query_words=set(query.lower().split())
        
        scored_examples=[]
        for ex in self.examples:
            ex_words=set(ex.input.lower().split())
            similarity=len(query_words&ex_words)/max(len(query_words|ex_words),1)
            scored_examples.append((similarity*ex.quality,ex))
        
        scored_examples.sort(reverse=True)
        return[ex for _,ex in scored_examples[:k]]
    
    def format_few_shot_prompt(self,examples:List[FewShotExample],query:str)->str:
        """Few-shotプロンプトをフォーマット"""
        prompt="Here are some examples:\n\n"
        for i,ex in enumerate(examples,1):
            prompt+=f"Example {i}:\nInput: {ex.input}\nOutput: {ex.output}\n\n"
        prompt+=f"Now, for this input:\nInput: {query}\nOutput:"
        return prompt

class SelfPlaySystem:
    """自己対戦学習システム"""
    def __init__(self):
        self.games:List[SelfPlayGame]=[]
        self.win_rates:Dict[str,float]=defaultdict(lambda:0.5)
        self.strategies:Dict[str,Dict]=defaultdict(dict)
    
    def create_game(self,question:str)->SelfPlayGame:
        """新しいゲームを作成"""
        game_id=str(uuid.uuid4())[:8]
        return SelfPlayGame(id=game_id,question=question,agent_a_answer="",agent_b_answer="")
    
    def play_game(self,game:SelfPlayGame,answer_a:str,answer_b:str):
        """ゲームをプレイ"""
        game.agent_a_answer=answer_a
        game.agent_b_answer=answer_b
        self.games.append(game)
    
    def judge_winner(self,game:SelfPlayGame,reasoning:str)->str:
        """勝者を判定"""
        # 簡易的な判定(実際はLLMで評価)
        if len(game.agent_a_answer)>len(game.agent_b_answer):
            winner="A"
        else:
            winner="B"
        
        game.winner=winner
        game.judge_reasoning=reasoning
        
        # 勝率更新
        self.win_rates["A"]=(self.win_rates["A"]*len(self.games)+float(winner=="A"))/(len(self.games)+1)
        
        return winner
    
    def get_best_strategy(self)->Dict:
        """最良の戦略を取得"""
        if not self.games:return{}
        
        # 勝ちゲームから戦略を抽出
        winning_games=[g for g in self.games if g.winner=="A"]
        if not winning_games:return{}
        
        return{
            'avg_length':statistics.mean(len(g.agent_a_answer)for g in winning_games),
            'success_rate':len(winning_games)/len(self.games),
            'patterns':[]  # 実際はパターン抽出が必要
        }

class MetaLearner:
    """メタ学習システム - 学習の学習"""
    def __init__(self):
        self.learning_curves:Dict[str,List[float]]=defaultdict(list)
        self.optimal_strategies:Dict[str,str]={}
        self.transfer_matrix:np.ndarray=np.zeros((10,10))
        self.task_embeddings:Dict[str,np.ndarray]={}
    
    def record_learning(self,task:str,performance:float):
        """学習曲線を記録"""
        self.learning_curves[task].append(performance)
    
    def estimate_learning_rate(self,task:str)->float:
        """学習率を推定"""
        curve=self.learning_curves.get(task,[])
        if len(curve)<2:return 0.1
        
        # 最近の改善率を計算
        recent=curve[-5:]
        if len(recent)<2:return 0.1
        
        improvements=[recent[i+1]-recent[i]for i in range(len(recent)-1)]
        return statistics.mean(improvements)if improvements else 0.1
    
    def suggest_strategy(self,task:str,current_performance:float)->str:
        """タスクに最適な戦略を提案"""
        if task in self.optimal_strategies:
            return self.optimal_strategies[task]
        
        # 類似タスクから推奨
        similar_tasks=self._find_similar_tasks(task)
        if similar_tasks:
            return self.optimal_strategies.get(similar_tasks[0],"balanced")
        
        # パフォーマンスに基づく推奨
        if current_performance<0.5:
            return"exploration"  # 探索重視
        elif current_performance>0.8:
            return"exploitation"  # 活用重視
        else:
            return"balanced"
    
    def _find_similar_tasks(self,task:str,top_k:int=3)->List[str]:
        """類似タスクを発見"""
        if task not in self.task_embeddings:
            return[]
        
        task_emb=self.task_embeddings[task]
        similarities=[]
        
        for other_task,other_emb in self.task_embeddings.items():
            if other_task==task:continue
            sim=np.dot(task_emb,other_emb)/(np.linalg.norm(task_emb)*np.linalg.norm(other_emb))
            similarities.append((sim,other_task))
        
        similarities.sort(reverse=True)
        return[t for _,t in similarities[:top_k]]
    
    def update_transfer_knowledge(self,source_task:str,target_task:str,improvement:float):
        """転移学習の効果を記録"""
        # 簡易的な転移行列更新
        source_idx=hash(source_task)%10
        target_idx=hash(target_task)%10
        self.transfer_matrix[source_idx,target_idx]=improvement

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
        
        # Ultra-advanced components
        self.tot:Optional[TreeOfThoughts]=TreeOfThoughts()if self.cfg.tree_of_thoughts else None
        self.debate:Optional[DebateSystem]=DebateSystem()if self.cfg.debate else None
        self.critic:Optional[CriticSystem]=CriticSystem()if self.cfg.critic else None
        self.calibrator:Optional[ConfidenceCalibrator]=ConfidenceCalibrator()if self.cfg.confidence_calibration else None
        self.curriculum:Optional[CurriculumLearner]=CurriculumLearner()if self.cfg.curriculum else None
        self.active_queries:List[ActiveLearningQuery]=[]if self.cfg.active_learning else None
        
        # Next-gen components
        self.constitutional:Optional[ConstitutionalAI]=ConstitutionalAI()if self.cfg.constitutional_ai else None
        self.world_model:Optional[WorldModel]=WorldModel()if self.cfg.world_model else None
        self.causal:Optional[CausalReasoning]=CausalReasoning()if self.cfg.causal_reasoning else None
        self.analogy:Optional[AnalogyEngine]=AnalogyEngine()if self.cfg.analogy_engine else None
        self.few_shot:Optional[FewShotLearner]=FewShotLearner()if self.cfg.few_shot_learning else None
        self.self_play:Optional[SelfPlaySystem]=SelfPlaySystem()if self.cfg.self_play else None
        self.meta_learner:Optional[MetaLearner]=MetaLearner()if self.cfg.meta_learning else None
        
        log.l.info(f"✅ Ultra-Advanced Init: {self.cfg.model}")
        features=[f"🧠Adapt:{self.cfg.adapt}",f"🎰MAB:{self.cfg.mab}",
                 f"💾Memory:{self.cfg.memory}",f"🧩KG:{self.cfg.kg}",
                 f"🤔CoT:{self.cfg.cot}",f"🔄Reflect:{self.cfg.reflection}",
                 f"🎲Thompson:{self.cfg.thompson}",f"📊AB:{self.cfg.ab_test}",
                 f"🎭Ensemble:{self.cfg.ensemble}",f"🧘MetaCog:{self.cfg.metacog}"]
        
        ultra_features=[f"🌳ToT:{self.cfg.tree_of_thoughts}",f"🗣️Debate:{self.cfg.debate}",
                       f"🔍Critic:{self.cfg.critic}",f"📏Calib:{self.cfg.confidence_calibration}",
                       f"📚Curric:{self.cfg.curriculum}",f"🎯Active:{self.cfg.active_learning}"]
        
        nextgen_features=[f"⚖️Const:{self.cfg.constitutional_ai}",f"🌍World:{self.cfg.world_model}",
                         f"🔗Causal:{self.cfg.causal_reasoning}",f"🎭Analogy:{self.cfg.analogy_engine}",
                         f"📝Few-shot:{self.cfg.few_shot_learning}",f"♟️Self-play:{self.cfg.self_play}",
                         f"🧬Meta:{self.cfg.meta_learning}"]
        
        log.l.info(" | ".join(features))
        log.l.info("🚀 Ultra: "+(" | ".join(ultra_features)))
        log.l.info("🌟 Next-Gen: "+(" | ".join(nextgen_features)))
    
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
    
    async def _execute_tree_of_thoughts(self,query:str,model:str,temp:float,max_tok:int)->Resp:
        """Tree of Thoughts実行"""
        if not self.tot:
            return await self._execute_direct(query,model,temp,max_tok)
        
        # ルートノード作成
        root_id=self.tot.create_root(query)
        
        # 複数の思考パスを探索
        for depth in range(self.tot.max_depth):
            # 現在の葉ノードを取得
            leaf_nodes=[nid for nid,node in self.tot.nodes.items() 
                       if not node.children and node.depth==depth]
            
            for node_id in leaf_nodes[:3]:  # 最大3ノード展開
                node=self.tot.nodes[node_id]
                
                # 次の思考候補を生成
                prompt=f"Given the thought: '{node.content}'\nGenerate 3 different next steps of reasoning:"
                ar=await self._api_call(model,[
                    {"role":"system","content":"You are a logical reasoning expert."},
                    {"role":"user","content":prompt}
                ],temp,max_tok//2)
                
                # 候補を抽出
                text=ar.choices[0].message.content or""
                candidates=re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)',text,re.DOTALL)
                candidates=[c.strip()for c in candidates if c.strip()]
                
                # ノード展開
                self.tot.expand(node_id,candidates)
                
                # 各候補を評価
                for child_id in self.tot.nodes[node_id].children:
                    child=self.tot.nodes[child_id]
                    eval_prompt=f"Rate the quality of this reasoning step (0-1): '{child.content}'"
                    eval_ar=await self._api_call(model,[
                        {"role":"system","content":"Rate between 0 and 1."},
                        {"role":"user","content":eval_prompt}
                    ],0.3,50)
                    
                    eval_text=eval_ar.choices[0].message.content or"0.5"
                    try:
                        value=float(re.search(r'0?\.\d+|[01]',eval_text).group())
                    except:
                        value=0.5
                    
                    # 逆伝播
                    self.tot.backpropagate(child_id,value)
        
        # 最良パスを選択
        best_path=self.tot.select_best_path()
        path_content=self.tot.get_path_content(best_path)
        
        # 最終回答を生成
        final_prompt=f"Based on this reasoning path:\n"+"\n".join(f"{i+1}. {c}"for i,c in enumerate(path_content))+f"\n\nProvide a final answer to: {query}"
        
        final_ar=await self._api_call(model,[
            {"role":"system","content":"Synthesize the reasoning into a final answer."},
            {"role":"user","content":final_prompt}
        ],temp,max_tok)
        
        resp=self._build_response(final_ar,model,Strategy.COT)
        resp.reasoning_steps=path_content
        return resp
    
    async def _execute_debate(self,query:str,model:str,temp:float,max_tok:int)->Resp:
        """討論モード実行"""
        if not self.debate:
            return await self._execute_direct(query,model,temp,max_tok)
        
        positions=["proponent","opponent","neutral"]
        
        for pos in positions:
            prompt=f"From the {pos} perspective, answer: {query}"
            ar=await self._api_call(model,[
                {"role":"system","content":f"You are a {pos} in a debate."},
                {"role":"user","content":prompt}
            ],temp,max_tok//3)
            
            text=ar.choices[0].message.content or""
            confidence=0.7+np.random.random()*0.2
            
            arg=DebateArgument(
                position=pos,argument=text,confidence=confidence
            )
            self.debate.add_argument(arg)
        
        # 統合
        synthesis=self.debate.synthesize()
        
        # 最終回答
        final_prompt=f"Synthesize these perspectives:\n{synthesis}\n\nProvide a balanced final answer to: {query}"
        final_ar=await self._api_call(model,[
            {"role":"system","content":"Provide a balanced synthesis."},
            {"role":"user","content":final_prompt}
        ],temp,max_tok)
        
        resp=self._build_response(final_ar,model,Strategy.ENSEMBLE)
        resp.alternatives=[arg.argument[:100]for arg in self.debate.arguments]
        return resp
    
    async def _execute_with_critic(self,query:str,model:str,temp:float,max_tok:int)->Resp:
        """批評システム付き実行"""
        if not self.critic:
            return await self._execute_direct(query,model,temp,max_tok)
        
        # 初期回答
        initial=await self._execute_direct(query,model,temp,max_tok)
        
        # 批評
        scores=self.critic.evaluate(initial.text,query)
        
        # 改善が必要な側面を特定
        weak_aspects=[asp for asp,score in scores.items()if score<0.6]
        
        if weak_aspects:
            improve_prompt=f"Improve the following answer, focusing on {', '.join(weak_aspects)}:\n\n{initial.text}\n\nOriginal question: {query}"
            
            improved_ar=await self._api_call(model,[
                {"role":"system","content":"Improve the answer based on feedback."},
                {"role":"user","content":improve_prompt}
            ],temp,max_tok)
            
            resp=self._build_response(improved_ar,model,Strategy.REFLECTION)
            resp.reflection=f"Initial (scores: {scores})"
            
            # フィードバック記録
            for aspect,score in scores.items():
                feedback=CriticFeedback(
                    aspect=aspect,score=score,
                    comments=f"{'Improved'if score>0.6 else'Needs work'}",
                    severity='low'if score>0.7 else'medium'
                )
                self.critic.add_feedback(feedback)
            
            return resp
        
        return initial
    
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
            
            # Ultra-advanced strategies
            if self.cfg.tree_of_thoughts and complexity==Complexity.EXPERT:
                tot_r=await self._execute_tree_of_thoughts(p,m,temp,mt)
                if tot_r.conf>r.conf:r=tot_r
            
            if self.cfg.debate and intent==Intent.ANALYSIS:
                debate_r=await self._execute_debate(p,m,temp,mt)
                if debate_r.conf>r.conf:r=debate_r
            
            if self.cfg.critic:
                r=await self._execute_with_critic(p,m,temp,mt)
            
            # 信頼度キャリブレーション
            if self.calibrator:
                r.conf=self.calibrator.calibrate(r.conf)
            
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
            
            # カリキュラム学習
            if self.curriculum:
                topic=intent.value if intent else"general"
                success=r.conf>0.7
                self.curriculum.update_mastery(topic,success,complexity.value)
            
            # 能動学習: 不確実性が高い場合はクエリを記録
            if self.cfg.active_learning and r.uncertainty>0.6:
                al_query=ActiveLearningQuery(
                    query=p[:200],uncertainty=r.uncertainty,
                    expected_gain=r.uncertainty*0.5,
                    priority=r.uncertainty*complexity.value
                )
                self.active_queries.append(al_query)
                if len(self.active_queries)>100:
                    self.active_queries=sorted(
                        self.active_queries,key=lambda x:x.priority,reverse=True
                    )[:100]
            
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
            
            # 信頼度キャリブレーション更新
            if self.calibrator and resp_obj:
                correct=rating>0
                self.calibrator.add_prediction(resp_obj.conf,correct)
            
            # カリキュラム学習更新
            if self.curriculum and intent:
                success=rating>0
                difficulty=0.3 if complexity==Complexity.SIMPLE else 0.5 if complexity==Complexity.MEDIUM else 0.7 if complexity==Complexity.COMPLEX else 0.9
                self.curriculum.update_mastery(intent.value,success,difficulty)
            
            log.l.info(f"🧠 Deep Learning: {rating:+d} | {intent} | {complexity} | {strategy}")
    
    def get_active_learning_suggestions(self,top_k:int=5)->List[str]:
        """能動学習の推奨質問を取得"""
        if not self.cfg.active_learning or not self.active_queries:
            return[]
        
        # 優先度でソート
        sorted_queries=sorted(self.active_queries,key=lambda x:x.priority,reverse=True)
        unanswered=[q.query for q in sorted_queries if not q.asked]
        return unanswered[:top_k]
    
    def get_curriculum_status(self)->Dict[str,Any]:
        """カリキュラム学習の状態を取得"""
        if not self.curriculum:return{}
        
        return{
            'mastery_levels':dict(self.curriculum.mastery),
            'topics_need_review':[t for t in self.curriculum.mastery.keys()
                                 if self.curriculum.should_review(t)],
            'recommended_difficulties':{t:self.curriculum.get_next_difficulty(t)
                                       for t in self.curriculum.mastery.keys()}
        }
    
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
        
        # Ultra-advanced stats
        if self.calibrator:
            base_stats['calibration']={
                'ece':f"{self.calibrator.get_calibration_error():.3f}",
                'predictions':len(self.calibrator.predictions)
            }
        
        if self.curriculum:
            curric_stats=self.get_curriculum_status()
            base_stats['curriculum']={
                'topics':len(curric_stats.get('mastery_levels',{})),
                'needs_review':curric_stats.get('topics_need_review',[])[:3],
                'avg_mastery':statistics.mean(curric_stats.get('mastery_levels',{0.5:0.5}).values())if curric_stats.get('mastery_levels')else 0.5
            }
        
        if self.cfg.active_learning and self.active_queries:
            base_stats['active_learning']={
                'pending_queries':len([q for q in self.active_queries if not q.asked]),
                'avg_uncertainty':statistics.mean(q.uncertainty for q in self.active_queries)if self.active_queries else 0,
                'top_priority':self.active_queries[0].query[:50]if self.active_queries else None
            }
        
        if self.critic:
            base_stats['critic']={
                'feedbacks':len(self.critic.feedbacks),
                'overall_score':f"{self.critic.get_overall_score():.2f}"
            }
        
        if self.tot:
            base_stats['tree_of_thoughts']={
                'total_nodes':len(self.tot.nodes),
                'max_depth':self.tot.max_depth
            }
        
        if self.debate:
            base_stats['debate']={
                'arguments':len(self.debate.arguments),
                'positions':list(self.debate.positions.keys())
            }
        
        # Next-gen stats
        if self.constitutional:
            total_applied=sum(r.applied for r in self.constitutional.rules)
            total_violations=sum(r.violations for r in self.constitutional.rules)
            base_stats['constitutional']={
                'rules':len(self.constitutional.rules),
                'compliance_rate':f"{1.0-total_violations/max(total_applied,1):.1%}",
                'violations':total_violations
            }
        
        if self.world_model:
            base_stats['world_model']={
                'states':len(self.world_model.states),
                'transitions':len(self.world_model.transitions),
                'beliefs':len(self.world_model.current_state.beliefs)
            }
        
        if self.causal:
            base_stats['causal']={
                'links':len(self.causal.causal_links),
                'avg_strength':statistics.mean(l.strength for l in self.causal.causal_links)if self.causal.causal_links else 0
            }
        
        if self.analogy:
            base_stats['analogy']={
                'analogies':len(self.analogy.analogies),
                'domains':len(self.analogy.domain_knowledge)
            }
        
        if self.few_shot:
            base_stats['few_shot']={
                'examples':len(self.few_shot.examples),
                'avg_quality':statistics.mean(e.quality for e in self.few_shot.examples)if self.few_shot.examples else 0
            }
        
        if self.self_play:
            base_stats['self_play']={
                'games':len(self.self_play.games),
                'win_rate_a':f"{self.self_play.win_rates.get('A',0.5):.1%}"
            }
        
        if self.meta_learner:
            base_stats['meta_learning']={
                'tasks':len(self.meta_learner.learning_curves),
                'total_records':sum(len(curve)for curve in self.meta_learner.learning_curves.values())
            }
        
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
        print("  /stats      - 統計情報表示")
        print("  /save       - データ保存")
        print("  /load       - データ読み込み")
        print("  /feedback <rating> - 最後の回答に評価 (-2 to +2)")
        print("  /clear      - 履歴クリア")
        print("  /suggest    - 能動学習の推奨質問")
        print("  /curriculum - 学習進捗状況")
        print("  /calibration- 信頼度キャリブレーション状態")
        print("  /causal     - 因果推論ネットワーク")
        print("  /analogy    - アナロジー知識ベース")
        print("  /world      - 世界モデル状態")
        print("  /selfplay   - 自己対戦学習状態")
        print("  /meta       - メタ学習進捗")
        print("  /help       - ヘルプ表示")
        print("  /exit       - 終了")
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
        
        # Ultra-advanced stats
        if 'calibration' in stats:
            cal=stats['calibration']
            print(f"\n📏 Calibration: ECE={cal['ece']} | Predictions={cal['predictions']}")
        
        if 'curriculum' in stats:
            cur=stats['curriculum']
            print(f"\n📚 Curriculum: {cur['topics']} topics | Avg mastery={cur['avg_mastery']:.2f}")
            if cur['needs_review']:
                print(f"   Review needed: {', '.join(cur['needs_review'])}")
        
        if 'active_learning' in stats:
            al=stats['active_learning']
            print(f"\n🎯 Active Learning: {al['pending_queries']} pending | Avg uncertainty={al['avg_uncertainty']:.2f}")
            if al['top_priority']:
                print(f"   Top: {al['top_priority']}")
        
        if 'critic' in stats:
            cri=stats['critic']
            print(f"\n🔍 Critic: {cri['feedbacks']} feedbacks | Score={cri['overall_score']}")
        
        if 'tree_of_thoughts' in stats:
            tot=stats['tree_of_thoughts']
            print(f"\n🌳 ToT: {tot['total_nodes']} nodes | Max depth={tot['max_depth']}")
        
        if 'debate' in stats:
            deb=stats['debate']
            print(f"\n🗣️ Debate: {deb['arguments']} arguments | Positions={', '.join(deb['positions'])}")
        
        # Next-gen stats
        if 'constitutional' in stats:
            const=stats['constitutional']
            print(f"\n⚖️ Constitutional: {const['rules']} rules | Compliance={const['compliance_rate']} | Violations={const['violations']}")
        
        if 'world_model' in stats:
            wm=stats['world_model']
            print(f"\n🌍 World Model: {wm['states']} states | {wm['transitions']} transitions | {wm['beliefs']} beliefs")
        
        if 'causal' in stats:
            caus=stats['causal']
            print(f"\n🔗 Causal: {caus['links']} links | Avg strength={caus['avg_strength']:.2f}")
        
        if 'analogy' in stats:
            ana=stats['analogy']
            print(f"\n🎭 Analogy: {ana['analogies']} analogies | {ana['domains']} domains")
        
        if 'few_shot' in stats:
            fs=stats['few_shot']
            print(f"\n📝 Few-shot: {fs['examples']} examples | Avg quality={fs['avg_quality']:.2f}")
        
        if 'self_play' in stats:
            sp=stats['self_play']
            print(f"\n♟️ Self-play: {sp['games']} games | Win rate A={sp['win_rate_a']}")
        
        if 'meta_learning' in stats:
            ml=stats['meta_learning']
            print(f"\n🧬 Meta-learning: {ml['tasks']} tasks | {ml['total_records']} records")
        
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
        
        elif command=='/suggest':
            # 能動学習の推奨質問
            suggestions=self.llm.get_active_learning_suggestions(5)
            if suggestions:
                print("\n🎯 Suggested questions to improve the system:")
                for i,s in enumerate(suggestions,1):
                    print(f"  {i}. {s}")
            else:
                print("ℹ️  No suggestions available")
        
        elif command=='/curriculum':
            # カリキュラム状態
            status=self.llm.get_curriculum_status()
            if status:
                print("\n📚 Learning Progress:")
                for topic,mastery in sorted(status.get('mastery_levels',{}).items(),
                                           key=lambda x:x[1],reverse=True):
                    bar='█'*int(mastery*20)+'░'*(20-int(mastery*20))
                    print(f"  {topic:15s} [{bar}] {mastery:.2%}")
                
                if status.get('topics_need_review'):
                    print(f"\n  📖 Review: {', '.join(status['topics_need_review'][:5])}")
            else:
                print("ℹ️  Curriculum learning not enabled")
        
        elif command=='/calibration':
            # キャリブレーション情報
            if self.llm.calibrator:
                ece=self.llm.calibrator.get_calibration_error()
                print(f"\n📏 Confidence Calibration:")
                print(f"  Expected Calibration Error: {ece:.3f}")
                print(f"  Predictions tracked: {len(self.llm.calibrator.predictions)}")
                if ece<0.05:
                    print("  ✅ Well calibrated!")
                elif ece<0.15:
                    print("  ⚠️  Moderate miscalibration")
                else:
                    print("  ❌ Poor calibration - confidence scores unreliable")
            else:
                print("ℹ️  Calibration not enabled")
        
        elif command=='/causal':
            # 因果推論状態
            if self.llm.causal:
                print(f"\n🔗 Causal Reasoning Network:")
                print(f"  Total causal links: {len(self.llm.causal.causal_links)}")
                
                if self.llm.causal.causal_links:
                    print("\n  Top causal relationships:")
                    sorted_links=sorted(self.llm.causal.causal_links,
                                       key=lambda l:l.strength,reverse=True)[:5]
                    for i,link in enumerate(sorted_links,1):
                        print(f"    {i}. {link.cause[:30]} → {link.effect[:30]} "
                              f"(strength: {link.strength:.2f})")
            else:
                print("ℹ️  Causal reasoning not enabled")
        
        elif command=='/analogy':
            # アナロジー状態
            if self.llm.analogy:
                print(f"\n🎭 Analogy Engine:")
                print(f"  Domains: {len(self.llm.analogy.domain_knowledge)}")
                print(f"  Analogies found: {len(self.llm.analogy.analogies)}")
                
                if self.llm.analogy.domain_knowledge:
                    print("\n  Knowledge domains:")
                    for domain,knowledge in list(self.llm.analogy.domain_knowledge.items())[:5]:
                        print(f"    • {domain}: {len(knowledge)} facts")
            else:
                print("ℹ️  Analogy engine not enabled")
        
        elif command=='/world':
            # 世界モデル状態
            if self.llm.world_model:
                wm=self.llm.world_model
                print(f"\n🌍 World Model:")
                print(f"  Total states: {len(wm.states)}")
                print(f"  Transitions: {len(wm.transitions)}")
                print(f"  Current beliefs: {len(wm.current_state.beliefs)}")
                
                if wm.current_state.beliefs:
                    print("\n  Top beliefs:")
                    sorted_beliefs=sorted(wm.current_state.beliefs.items(),
                                         key=lambda x:x[1],reverse=True)[:5]
                    for belief,conf in sorted_beliefs:
                        bar='█'*int(conf*10)+'░'*(10-int(conf*10))
                        print(f"    [{bar}] {belief[:50]}")
            else:
                print("ℹ️  World model not enabled")
        
        elif command=='/selfplay':
            # 自己対戦状態
            if self.llm.self_play:
                sp=self.llm.self_play
                print(f"\n♟️ Self-Play System:")
                print(f"  Games played: {len(sp.games)}")
                print(f"  Agent A win rate: {sp.win_rates.get('A',0.5):.1%}")
                
                if sp.games:
                    print("\n  Recent games:")
                    for game in sp.games[-3:]:
                        winner_mark="🏆"if game.winner=="A"else"🥈"
                        print(f"    {winner_mark} {game.question[:50]}")
                        if game.judge_reasoning:
                            print(f"       Reason: {game.judge_reasoning[:60]}")
            else:
                print("ℹ️  Self-play not enabled")
        
        elif command=='/meta':
            # メタ学習状態
            if self.llm.meta_learner:
                ml=self.llm.meta_learner
                print(f"\n🧬 Meta-Learning:")
                print(f"  Tasks tracked: {len(ml.learning_curves)}")
                
                if ml.learning_curves:
                    print("\n  Learning progress:")
                    for task,curve in list(ml.learning_curves.items())[:5]:
                        if curve:
                            latest=curve[-1]
                            trend="📈"if len(curve)>1 and curve[-1]>curve[-2]else"📉"if len(curve)>1 else"➡️"
                            print(f"    {trend} {task}: {latest:.2%} "
                                  f"(learning rate: {ml.estimate_learning_rate(task):.3f})")
            else:
                print("ℹ️  Meta-learning not enabled")
        
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
    
    # Ultra-advanced options
    parser.add_argument('--enable-tot',action='store_true',
                       help='Enable Tree of Thoughts')
    parser.add_argument('--enable-debate',action='store_true',
                       help='Enable debate mode')
    parser.add_argument('--enable-critic',action='store_true',
                       help='Enable critic system')
    parser.add_argument('--enable-all-ultra',action='store_true',
                       help='Enable all ultra-advanced features')
    
    # Next-gen options
    parser.add_argument('--enable-constitutional',action='store_true',
                       help='Enable constitutional AI')
    parser.add_argument('--enable-causal',action='store_true',
                       help='Enable causal reasoning')
    parser.add_argument('--enable-analogy',action='store_true',
                       help='Enable analogy engine')
    parser.add_argument('--enable-world-model',action='store_true',
                       help='Enable world model')
    parser.add_argument('--enable-self-play',action='store_true',
                       help='Enable self-play learning')
    parser.add_argument('--enable-meta',action='store_true',
                       help='Enable meta-learning')
    parser.add_argument('--enable-all-nextgen',action='store_true',
                       help='Enable ALL next-gen features')
    
    args=parser.parse_args()
    
    # Ultra機能の有効化
    enable_ultra=args.enable_all_ultra
    enable_nextgen=args.enable_all_nextgen or args.enable_all_ultra
    
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
        thompson=not args.no_thompson,
        # Ultra-advanced
        tree_of_thoughts=args.enable_tot or enable_ultra,
        debate=args.enable_debate or enable_ultra,
        critic=args.enable_critic or enable_ultra,
        confidence_calibration=enable_ultra,
        active_learning=enable_ultra,
        curriculum=enable_ultra,
        # Next-gen
        constitutional_ai=args.enable_constitutional or enable_nextgen,
        causal_reasoning=args.enable_causal or enable_nextgen,
        analogy_engine=args.enable_analogy or enable_nextgen,
        world_model=args.enable_world_model or enable_nextgen,
        self_play=args.enable_self_play or enable_nextgen,
        meta_learning=args.enable_meta or enable_nextgen,
        few_shot_learning=enable_nextgen
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
