import json
import pickle
from pathlib import Path
import numpy as np
import pytest

from owe.sequential_retriever import HybridEngine, prepare_contexts, masked_contexts, block_fold_sources
from owe.prepare import stable_int
from owe.sequential_interventions import batch_rank,make_case
from owe.sequential_retriever import model_scores


@pytest.fixture(scope="module")
def study():
    path = Path("data/processed/scoped/dataset.pkl")
    if not path.exists():
        pytest.skip("Prepared research data required")
    config = json.loads(Path("configs/sequential_v2.json").read_text())
    data = pickle.loads(path.read_bytes())
    vectorizer = pickle.loads(Path("runs/scoped_v1/vectorizer.pkl").read_bytes())
    contexts = prepare_contexts(pickle.loads(Path("runs/scoped_v1/graphs.pkl").read_bytes()))
    return config, data, contexts, HybridEngine(data, vectorizer, config)


def test_candidate_generation_needs_no_gold_or_corruption_labels(study):
    config, data, contexts, engine = study
    queries = [q for q in data["queries"] if q["split"] == "train"]
    for query in queries[::max(1, len(queries) // 8)]:
        original = engine.retrieve(query, contexts["random20_s17"], config["candidate_variants"][1])
        blind_query = {k: v for k, v in query.items() if k != "gold"}
        blind_context = {k: v for k, v in contexts["random20_s17"].items() if k != "audit_changes"}
        blind = engine.retrieve(blind_query, blind_context, config["candidate_variants"][1])
        np.testing.assert_array_equal(original.ids, blind.ids)
        np.testing.assert_array_equal(original.features, blind.features)
        assert len(blind.ids) <= config["candidate_cap"]
        assert all(int(u) < query["node"] for u in blind.ids)
        assert all(data["nodes"][int(u)]["signature"] != query["signature"] for u in blind.ids)
        engine.trim_cache()


def test_suppression_matches_materialized_deleted_graph(study):
    config, data, contexts, engine = study
    queries = [q for q in data["queries"] if q["split"] == "train" and q["topic"] == "graph"]
    for query in queries[::max(1, len(queries) // 6)][:6]:
        variant = config["candidate_variants"][1]
        context = contexts["random20_s17"]
        result = engine.retrieve(query, context, variant)
        deleted = list(result.edges[:3])
        if not deleted:
            continue
        suppressed = engine.retrieve(query, context, variant, deleted)
        edges = context["edges"] - set(deleted)
        adjacency = [[] for _ in data["nodes"]]
        incoming = [[] for _ in data["nodes"]]
        for u, v in sorted(edges):
            adjacency[u].append(v); incoming[v].append(u)
        fresh = prepare_contexts({"materialized": {"name": "materialized-" + query["qid"],
            "edges": edges, "adjacency": adjacency, "incoming": incoming}})["materialized"]
        actual = engine.retrieve(query, fresh, variant)
        np.testing.assert_array_equal(suppressed.ids, actual.ids)
        np.testing.assert_allclose(suppressed.features, actual.features, atol=1e-7, rtol=0)
        assert suppressed.edges == actual.edges


def test_oof_hides_entire_query_signature_groups(study):
    config, data, contexts, _ = study
    fold = 1
    sources = block_fold_sources(data, fold, config["oof_folds"], stable_int)
    masked = masked_contexts({"clean": contexts["clean"]}, sources, "/oof1")["clean"]
    assert sources
    assert not (set(u for u, _ in masked["edges"]) & sources)
    train = [q for q in data["queries"] if q["split"] == "train"]
    heldout = {q["group"] for q in train if stable_int(q["group"]) % config["oof_folds"] == fold}
    fitting = {q["group"] for q in train if stable_int(q["group"]) % config["oof_folds"] != fold}
    assert heldout.isdisjoint(fitting)


def test_exact_prediction_cache_matches_full_reranking(study):
    config,data,contexts,engine=study
    model_path=Path('runs/sequential_v2/ranker.pkl')
    if not model_path.exists(): pytest.skip('Fit the stage2 model first')
    ranker=pickle.loads(model_path.read_bytes())
    variant=json.loads(Path('runs/sequential_v2/stage1_selection.json').read_text())['variant']
    queries=[q for q in data['queries'] if q['split']=='train' and q['topic']=='graph']
    for q in queries[::max(1,len(queries)//5)][:5]:
        base=engine.retrieve(q,contexts['random20_s17'],variant)
        changes=[engine.retrieve(q,contexts['random20_s17'],variant,[edge]) for edge in base.edges[:5]]
        full=batch_rank(ranker,changes)
        cached=batch_rank(ranker,changes,base,model_scores(ranker,base.features))
        for a,b in zip(full,cached): np.testing.assert_array_equal(a,b)


def test_optimized_engine_matches_completed_oof_checkpoint(study):
    config,data,_,engine=study
    folder=Path('runs/sequential_v2')
    if not (folder/'oof0_intervention_cases.pkl').exists(): pytest.skip('Complete an OOF checkpoint first')
    cases=pickle.loads((folder/'oof0_intervention_cases.pkl').read_bytes())
    contexts=pickle.loads((folder/'oof0_contexts.pkl').read_bytes())
    model=pickle.loads((folder/'oof0_ranker.pkl').read_bytes())
    variant=json.loads((folder/'stage1_selection.json').read_text())['variant']
    queries={q['qid']:q for q in data['queries']}
    for case in cases[::max(1,len(cases)//5)][:5]:
        query=queries[case['qid']];context=contexts[case['condition']]
        base=engine.retrieve(query,context,variant)
        np.testing.assert_array_equal(base.ids,case['base_candidates'])
        np.testing.assert_array_equal(model_scores(model,base.features),case['base_scores'])


def test_edge_inference_needs_no_query_gold(study):
    from owe.sequential_interventions import edge_features,candidate_edges
    config,data,contexts,engine=study
    path=Path('runs/sequential_v2/ranker.pkl')
    if not path.exists():pytest.skip('Fit a ranker first')
    model=pickle.loads(path.read_bytes())
    q=next(q for q in data['queries'] if q['split']=='train' and q['topic']=='graph')
    variant=config['candidate_variants'][1];context=contexts['random20_s17']
    result=engine.retrieve(q,context,variant);p=model_scores(model,result.features)
    edges=candidate_edges(result,q,'random20_s17',p,24)
    actual=edge_features(engine,q,context,result,p,edges)
    blind=edge_features(engine,{k:v for k,v in q.items() if k!='gold'},context,result,p,edges)
    np.testing.assert_array_equal(actual,blind)


def test_cpu_worker_labels_equal_serial_labels(study):
    from owe.sequential_workers import generate
    config,data,contexts,engine=study
    path=Path('runs/sequential_v2/ranker.pkl')
    if not path.exists():pytest.skip('Fit a ranker first')
    ranker=pickle.loads(path.read_bytes())
    variant=json.loads(Path('runs/sequential_v2/stage1_selection.json').read_text())['variant']
    qs=[next(q for q in data['queries'] if q['split']=='train' and q['topic']==topic) for topic in ['graph','sets_relations_functions']]
    actual=generate(engine,{'clean':contexts['clean']},qs,variant,ranker,['clean'],'test-workers',workers=2)
    for query,case in zip(qs,actual):
        serial=make_case(engine,query,contexts['clean'],variant,ranker,'clean')
        assert serial['edges']==case['edges']
        for key in ['features','benefits','base_candidates','base_scores','base_ranking']:
            np.testing.assert_array_equal(serial[key],case[key])
        for a,b in zip(serial['single_rankings'],case['single_rankings']):np.testing.assert_array_equal(a,b)
