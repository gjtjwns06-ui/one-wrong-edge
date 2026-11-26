"""Deterministic CPU workers for independent intervention-label cases."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
from .sequential_retriever import HybridEngine
from .sequential_interventions import make_case

_ENGINE = _CONTEXTS = _VARIANT = _RANKER = _LIMITS = None


def initialize(config,data,contexts,vectorizer,variant,ranker):
    global _ENGINE,_CONTEXTS,_VARIANT,_RANKER,_LIMITS
    _LIMITS=threadpool_limits(limits=1)
    _ENGINE=HybridEngine(data,vectorizer,config)
    _CONTEXTS=contexts;_VARIANT=variant;_RANKER=ranker


def run_case(job):
    query,condition,fold=job
    result=make_case(_ENGINE,query,_CONTEXTS[condition],_VARIANT,_RANKER,condition,fold)
    _ENGINE.trim_cache()
    return result


def generate(engine,contexts,queries,variant,ranker,conditions,label,fold=None,workers=3):
    jobs=[(q,condition,fold) for condition in conditions for q in queries]
    cases=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,
        initargs=(engine.config,engine.data,contexts,engine.vectorizer,variant,ranker)) as executor:
        for index,case in enumerate(executor.map(run_case,jobs,chunksize=8)):
            cases.append(case)
            if (index+1)%200==0:
                print(f'{label}: {index+1}/{len(jobs)} intervention queries ({workers} CPU workers)',flush=True)
    return cases
