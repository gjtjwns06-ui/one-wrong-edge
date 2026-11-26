"""Deterministic independent fresh intervention cases on CPU workers."""
from concurrent.futures import ProcessPoolExecutor
import time
from threadpoolctl import threadpool_limits
from .sequential_retriever import HybridEngine
from .sequential_interventions import make_case
from .restart_support import query_context

_ENGINE=_CONTEXTS=_VARIANT=_RANKER=_LIMITS=None


def initialize(config,data,contexts,vectorizer,variant,ranker):
    global _ENGINE,_CONTEXTS,_VARIANT,_RANKER,_LIMITS
    _LIMITS=threadpool_limits(limits=1)
    _ENGINE=HybridEngine(data,vectorizer,config)
    _CONTEXTS=contexts;_VARIANT=variant;_RANKER=ranker


def run_case(job):
    started=time.perf_counter()
    query,condition=job
    context=query_context(_ENGINE,query,_CONTEXTS,condition,_VARIANT)
    case=make_case(_ENGINE,query,context,_VARIANT,_RANKER,condition)
    case['injected_edges']=len(context['edges']-_CONTEXTS['clean']['edges'])
    case['exposed_false_edges']=len(set(_ENGINE.retrieve(query,context,_VARIANT).edges)-_CONTEXTS['clean']['edges'])
    case['local_changes']=context.get('audit_changes',[]) if condition.startswith('local') else []
    case['offline_seconds']=time.perf_counter()-started
    _ENGINE.trim_cache()
    return case


def generate(engine,contexts,queries,variant,ranker,conditions,label):
    jobs=[(q,c) for c in conditions for q in queries]
    result=[]
    with ProcessPoolExecutor(max_workers=engine.config['workers'],initializer=initialize,
        initargs=(engine.config,engine.data,contexts,engine.vectorizer,variant,ranker)) as pool:
        for i,case in enumerate(pool.map(run_case,jobs,chunksize=8)):
            result.append(case)
            if (i+1)%200==0:print(f'{label}: {i+1}/{len(jobs)} cases',flush=True)
    return result
