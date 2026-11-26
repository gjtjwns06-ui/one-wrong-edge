"""OOF intervention supervision, policy selection and final sequential evaluation."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .experiment import write_json
from .prepare import stable_int
from .retrieval import metrics
from .sequential_retriever import block_fold_sources, masked_contexts, model_scores, predicted_ranking
from .sequential_stage12 import ROOT, fit_ranker, load_study, training_arrays
from .sequential_interventions import EDGE_FEATURES, make_case, utility_predictions, choose_actions, batch_rank,edge_features
from .sequential_workers import generate


def dump(path, obj):
    with Path(path).open("wb") as f:
        pickle.dump(obj, f, protocol=5)


def load(path):
    return pickle.loads(Path(path).read_bytes())


def oof(config, data, contexts, engine):
    variant = json.loads((ROOT / "stage1_selection.json").read_text())["variant"]
    name = json.loads((ROOT / "stage2_selection.json").read_text())["model"]
    train = [q for q in data["queries"] if q["split"] == "train"]
    folds = config["oof_folds"]
    for fold in range(folds):
        case_path = ROOT / f"oof{fold}_intervention_cases.pkl"
        if case_path.exists():
            print(f"OOF fold {fold}: retained existing completed cases", flush=True)
            continue
        start = time.perf_counter()
        sources = block_fold_sources(data, fold, folds, stable_int)
        fold_contexts = masked_contexts({k: contexts[k] for k in config["training_conditions"]}, sources, f"/oof{fold}")
        fitting = [q for q in train if stable_int(q["group"]) % folds != fold]
        heldout = [q for q in train if stable_int(q["group"]) % folds == fold]
        assert {q["group"] for q in fitting}.isdisjoint(q["group"] for q in heldout)
        model_path = ROOT / f"oof{fold}_ranker.pkl"
        if model_path.exists():
            ranker = load(model_path)
        else:
            x, y, w, records = training_arrays(engine, fold_contexts, fitting, variant, config, f"OOF{fold}")
            ranker = fit_ranker(name, config, x, y, w)
            dump(model_path, ranker)
            pd.DataFrame(records).to_csv(ROOT / f"oof{fold}_ranker_training_cases.csv", index=False)
            del x, y, w, records
        dump(ROOT / f"oof{fold}_contexts.pkl", fold_contexts)
        write_json(ROOT / f"oof{fold}_audit.json", {"fold": fold,
            "fitting_qids": [q["qid"] for q in fitting], "heldout_qids": [q["qid"] for q in heldout],
            "fitting_groups": sorted({q["group"] for q in fitting}), "heldout_groups": sorted({q["group"] for q in heldout}),
            "hidden_sources": sorted(sources), "ranker_name": name})
        cases=generate(engine,fold_contexts,heldout,variant,ranker,config['training_conditions'],f'OOF{fold}',fold)
        dump(case_path, cases)
        print(f"OOF fold {fold} complete: {len(cases)} cases, {time.perf_counter()-start:.1f}s", flush=True)


def fit_utility_models(cases, reference, config):
    x = np.concatenate([c["features"] for c in cases])
    y = np.concatenate([c["benefits"][:, 1] for c in cases])
    wrong = np.asarray([edge not in reference for c in cases for edge in c["edges"]], dtype=np.int8)
    weights = np.concatenate([np.full(len(c["edges"]), 1/max(1, len(c["edges"]))) for c in cases])
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(x, sample_weight=weights)
    models = {}
    for alpha in config["utility_ridge_alpha"]:
        estimator = Ridge(alpha=alpha).fit(scaler.transform(x), y, sample_weight=weights)
        models[f"ridge_{alpha:g}"] = {"kind": "utility", "estimator": estimator, "scaler": scaler}
    regressor = HistGradientBoostingRegressor(max_iter=config["utility_boost_iterations"],
        max_leaf_nodes=config["utility_boost_leaf_nodes"], learning_rate=0.08, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=config["random_seed"])
    regressor.fit(x, y, sample_weight=weights)
    models["utility_boost"] = {"kind": "utility", "estimator": regressor, "scaler": None}
    classification_weights = weights.copy()
    for label in [0, 1]:
        mask = wrong == label
        classification_weights[mask] *= weights.sum() / (2 * weights[mask].sum())
    classifier = HistGradientBoostingClassifier(max_iter=config["utility_boost_iterations"],
        max_leaf_nodes=config["utility_boost_leaf_nodes"], learning_rate=0.08, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=config["random_seed"])
    classifier.fit(x, wrong, sample_weight=classification_weights)
    models["wrongness_boost"] = {"kind": "wrongness", "estimator": classifier, "scaler": None}
    write_json(ROOT / "utility_training_summary.json", {"OOF_query_context_cases": len(cases),
        "candidate_effects": len(y), "positive": int((y > 1e-8).sum()), "negative": int((y < -1e-8).sum()),
        "zero": int((np.abs(y) <= 1e-8).sum()), "wrong_attribution_candidates": int(wrong.sum()),
        "edge_features": EDGE_FEATURES})
    np.savez_compressed(ROOT / "utility_training.npz", features=x, benefits=y, wrongness_labels=wrong, weights=weights)
    return models


def select_policies(cases, models, ranker, queries, engine, contexts, variant, config):
    specs = []
    for name, model in models.items():
        thresholds = config["wrongness_thresholds"] if model["kind"] == "wrongness" else config["utility_thresholds"]
        for budget in config["deletion_budgets"]:
            for threshold in thresholds:
                specs.append({"model": name, "kind": model["kind"], "budget": budget, "threshold": threshold})
    sums = np.zeros(len(specs)); counts = np.zeros(len(specs))
    for case_index, case in enumerate(cases):
        predictions = {name: utility_predictions(model, case["features"]) for name, model in models.items()}
        action_sets = [tuple(choose_actions(case, predictions[s["model"]], s["budget"], s["threshold"])) for s in specs]
        unique_batches = sorted({a for a in action_sets if len(a) > 1})
        query = queries[case["qid"]]
        base_result = engine.retrieve(query, contexts[case["condition"]], variant)
        results = [engine.retrieve(query, contexts[case["condition"]], variant, actions) for actions in unique_batches]
        ranks = batch_rank(ranker, results, base_result, case["base_scores"])
        base = metrics(case["base_ranking"], set(query["gold"]), 20)["recall"]
        effects = {actions: metrics(ids, set(query["gold"]), 20)["recall"] - base
                   for actions, ids in zip(unique_batches, ranks)}
        effects[()] = 0.0
        for edge_index, edge in enumerate(case["edges"]):
            effects[(edge,)] = float(case["benefits"][edge_index, 1])
        for index, actions in enumerate(action_sets):
            sums[index] += effects[actions]
            counts[index] += len(actions)
        engine.trim_cache()
        if (case_index + 1) % 250 == 0:
            print(f"Policy validation: {case_index + 1}/{len(cases)} cases", flush=True)
    records = [{**spec, "validation_gain": float(sums[i]/len(cases)),
                "validation_mean_deletions": float(counts[i]/len(cases))} for i, spec in enumerate(specs)]
    write_json(ROOT / "stage3_validation_grid.json", records)
    selected = {}
    for kind in ["utility", "wrongness"]:
        for budget in config["deletion_budgets"]:
            options = [r for r in records if r["kind"] == kind and r["budget"] == budget]
            selected[f"{kind}{budget}"] = max(options, key=lambda r: (round(r["validation_gain"], 12), -r["validation_mean_deletions"]))
    return selected


def select_filter(cases,model,ranker,queries,engine,contexts,variant):
    specs=[{'model':'wrongness_boost','kind':'wrongness','scope':'all_traversed','budget':budget,'threshold':threshold}
           for budget in [24,64,128] for threshold in [0.5,0.7,0.9,1.01]]
    sums=np.zeros(len(specs));counts=np.zeros(len(specs))
    for index,case in enumerate(cases):
        query=queries[case['qid']];context=contexts[case['condition']]
        base=engine.retrieve(query,context,variant)
        edges=list(base.edges)
        features=edge_features(engine,query,context,base,case['base_scores'],edges)
        scores=utility_predictions(model,features)
        actions=[tuple(choose_actions({'edges':edges},scores,s['budget'],s['threshold'])) for s in specs]
        unique=sorted(set(actions)-{()})
        changed=[engine.retrieve(query,context,variant,a) for a in unique]
        ranks=batch_rank(ranker,changed,base,case['base_scores'])
        baseline=metrics(case['base_ranking'],set(query['gold']),20)['recall']
        effects={a:metrics(ids,set(query['gold']),20)['recall']-baseline for a,ids in zip(unique,ranks)}
        effects[()]=0.0
        for j,a in enumerate(actions):sums[j]+=effects[a];counts[j]+=len(a)
        engine.trim_cache()
        if (index+1)%250==0:print(f'Filter validation: {index+1}/{len(cases)} cases',flush=True)
    rows=[{**s,'validation_gain':float(sums[i]/len(cases)),'validation_mean_deletions':float(counts[i]/len(cases))} for i,s in enumerate(specs)]
    write_json(ROOT/'filter_validation_grid.json',rows)
    return max(rows,key=lambda r:(round(r['validation_gain'],12),-r['validation_mean_deletions']))


def policy(config, data, contexts, engine):
    if (ROOT / "stage3_selection.json").exists():
        raise RuntimeError("Policy selection already frozen")
    start = time.perf_counter()
    variant = json.loads((ROOT / "stage1_selection.json").read_text())["variant"]
    ranker = load(ROOT / "ranker.pkl")
    training = []
    for fold in range(config["oof_folds"]):
        training.extend(load(ROOT / f"oof{fold}_intervention_cases.pkl"))
    models = fit_utility_models(training, set(map(tuple, data["edges"])), config)
    dump(ROOT / "utility_models.pkl", models)
    validation_path = ROOT / "utility_validation_cases.pkl"
    if validation_path.exists():
        validation = load(validation_path)
    else:
        queries = [q for q in data["queries"] if q["split"] == "validation"]
        validation=generate(engine,contexts,queries,variant,ranker,config['training_conditions'],'Utility validation')
        dump(validation_path, validation)
    queries = {q["qid"]: q for q in data["queries"]}
    selection = select_policies(validation, models, ranker, queries, engine, contexts, variant, config)
    selection['wrongness_filter']=select_filter(validation,models['wrongness_boost'],ranker,queries,engine,contexts,variant)
    write_json(ROOT / "stage3_selection.json", {"policies": selection, "seconds": time.perf_counter()-start,
        "stage2_selection_sha256": hashlib.sha256((ROOT / "stage2_selection.json").read_bytes()).hexdigest(),
        "selected_before_test": True})
    print("Stage3 policies locked: " + json.dumps(selection), flush=True)


def test(config, data, contexts, engine):
    if (ROOT / "run_complete.json").exists():
        raise RuntimeError("Completed sequential run already exists")
    start = time.perf_counter()
    variant = json.loads((ROOT / "stage1_selection.json").read_text())["variant"]
    ranker = load(ROOT / "ranker.pkl")
    models = load(ROOT / "utility_models.pkl")
    selection = json.loads((ROOT / "stage3_selection.json").read_text())["policies"]
    selections_hash = {f"stage{i}_selection.json": hashlib.sha256((ROOT/f"stage{i}_selection.json").read_bytes()).hexdigest() for i in [1,2,3]}
    write_json(ROOT / "test_lock.json", {"selection_hashes": selections_hash,
        "code_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("owe").glob("sequential*.py")},
        "intervention_workers":3,
        "test_reuse_is_exploratory": True})
    queries = [q for q in data["queries"] if q["split"] == "test"]
    old_rankings = {}
    with gzip.open(Path(config["source_run"]) / "retrievals.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            old_rankings[r["qid"], r["condition"]] = r["ranks"]["no_op"]
    reference = set(map(tuple, data["edges"]))
    clean_rankings = {}
    for condition, context in contexts.items():
        output_path = ROOT / f"test_{condition}.csv"
        ranks_path = ROOT / f"test_{condition}_rankings.jsonl.gz"
        if output_path.exists():
            if condition == "clean":
                with gzip.open(ranks_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line); clean_rankings[r["qid"]] = np.asarray(r["rankings"]["stage2_proof"], dtype=np.int32)
            print(f"Retained completed test condition {condition}", flush=True)
            continue
        case_path = ROOT / f"test_{condition}_cases.pkl"
        if case_path.exists():
            cases = load(case_path)
        else:
            cases=generate(engine,{condition:context},queries,variant,ranker,[condition],f'Test {condition}')
            dump(case_path, cases)
        rows = []; effect_rows = []
        with gzip.open(ranks_path, "wt", encoding="utf-8") as output:
            for query, case in zip(queries, cases):
                assert query["qid"] == case["qid"]
                result = engine.retrieve(query, context, variant)
                predictions = {name: utility_predictions(model, case["features"]) for name, model in models.items()}
                actions = {name: choose_actions(case, predictions[p["model"]], p["budget"], p["threshold"])
                           for name, p in selection.items() if p.get('scope')!='all_traversed'}
                if 'wrongness_filter' in selection:
                    p=selection['wrongness_filter'];filter_edges=list(result.edges)
                    if p['threshold']>=1:
                        actions['wrongness_filter']=[]
                    else:
                        filter_x=edge_features(engine,query,context,result,case['base_scores'],filter_edges)
                        filter_predictions=utility_predictions(models[p['model']],filter_x)
                        actions['wrongness_filter']=choose_actions({'edges':filter_edges},filter_predictions,p['budget'],p['threshold'])
                rng = np.random.default_rng(stable_int(query["qid"] + condition + "v2-matched-random"))
                order = rng.permutation(len(case["edges"]))
                for name in list(actions):
                    if name=='wrongness_filter':
                        pool=list(result.edges);filter_order=rng.permutation(len(pool))
                        actions['matched_'+name]=[pool[int(i)] for i in filter_order[:len(actions[name])]]
                    else:
                        actions["matched_"+name] = [case["edges"][int(i)] for i in order[:len(actions[name])]]
                actions["relevance3"] = sorted(case["edges"], key=lambda e: (float(engine.query_info(query)["scores"][e[1]]), e))[:3]
                if len(case["edges"]):
                    best = int(np.argmax(case["benefits"][:,1]))
                    actions["oracle1"] = [case["edges"][best]] if case["benefits"][best,1] > 0 else []
                else:
                    actions["oracle1"] = []
                rankings = {"stage2_proof": case["base_ranking"],
                            "v1_local": np.asarray(old_rankings[query["qid"], condition], dtype=np.int32),
                            "lexical": engine.query_info(query)["lexical"][:40],
                            "stage1_text": predicted_ranking(result, result.features[:,0])[:40],
                            "frequency": predicted_ranking(result, result.features[:,5])[:40]}
                hand = result.features[:,0] + 0.25*result.features[:,5]/max(1e-6,result.features[:,5].max())
                hand += 0.25*result.features[:,26]/max(1e-6,result.features[:,26].max())
                rankings["hand_mix"] = predicted_ranking(result, hand)[:40]
                if condition == "clean":
                    clean_rankings[query["qid"]] = case["base_ranking"]
                rankings["source_verified"] = clean_rankings[query["qid"]] if condition.startswith(("random", "hard")) else case["base_ranking"]
                known_singles=set(case['edges'])
                unique_batches = sorted({tuple(a) for a in actions.values() if len(a)>1 or (len(a)==1 and a[0] not in known_singles)})
                regenerated = [engine.retrieve(query, context, variant, a) for a in unique_batches]
                batch_results = dict(zip(unique_batches, batch_rank(ranker, regenerated, result, case["base_scores"])))
                single_map = {edge: ids for edge, ids in zip(case["edges"],case["single_rankings"])}
                for name, selected in actions.items():
                    rankings[name] = case["base_ranking"] if not selected else single_map[selected[0]] if len(selected)==1 and selected[0] in single_map else batch_results[tuple(selected)]
                gold = set(query["gold"])
                coverage = len(gold & set(result.ids)) / len(gold)
                ceiling = min(40,len(gold & set(result.ids))) / len(gold)
                for method, ids in rankings.items():
                    assert all(int(v) < query["node"] and engine.nodes[int(v)]["signature"] != query["signature"] for v in ids)
                    for budget in config["output_budgets"]:
                        measured = metrics(ids, gold, budget, engine.lengths)
                        base = metrics(case["base_ranking"], gold, budget)["recall"]
                        selected = actions.get(method, [])
                        rows.append({"qid":query["qid"],"group":query["group"],"topic":query["topic"],
                            "condition":condition,"method":method,"budget":budget,**measured,
                            "delta_vs_stage2":measured["recall"]-base,"deletions":len(selected),
                            "wrong_deletions":sum(edge not in reference for edge in selected),
                            "preprocessing_removed_edges":len(context["edges"]-reference) if method=="source_verified" else 0,
                            "gold_count":len(gold),"candidate_coverage":coverage,
                            "candidate_ceiling":min(budget,len(gold & set(result.ids)))/len(gold),
                            "candidates":len(result.ids),"transitions":result.transitions})
                for edge_index, edge in enumerate(case["edges"]):
                    effect_rows.append({"qid":query["qid"],"group":query["group"],"condition":condition,
                        "source":edge[0],"target":edge[1],"source_supported":int(edge in reference),
                        "benefit10":float(case["benefits"][edge_index,0]),"benefit20":float(case["benefits"][edge_index,1]),
                        "benefit40":float(case["benefits"][edge_index,2]),
                        "wrongness_prediction":float(predictions["wrongness_boost"][edge_index])})
                output.write(json.dumps({"qid":query["qid"],"condition":condition,"actions":actions,
                    "candidates":list(map(int,result.ids)),"rankings":{m:list(map(int,v)) for m,v in rankings.items()}})+"\n")
                engine.trim_cache()
        pd.DataFrame(rows).to_csv(output_path,index=False)
        pd.DataFrame(effect_rows).to_csv(ROOT/f"test_{condition}_effects.csv.gz",index=False)
        print(f"Completed test condition {condition}: {len(rows)} rows",flush=True)
    for filename, expected in selections_hash.items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected
    result_files=[ROOT/f"test_{name}.csv" for name in contexts]
    combined=pd.concat([pd.read_csv(p) for p in result_files],ignore_index=True)
    combined.to_csv(ROOT/"results.csv",index=False)
    write_json(ROOT/"run_complete.json",{"test_queries":len(queries),"conditions":len(contexts),
        "methods":combined.method.nunique(),"measurement_rows":len(combined),"test_seconds":time.perf_counter()-start,
        "selection_hashes":selections_hash,"test_reuse_is_exploratory":True})
    print("Sequential V2 completed",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["oof","policy","test"])
    args=parser.parse_args()
    config,data,contexts,engine=load_study()
    if args.stage=="oof": oof(config,data,contexts,engine)
    elif args.stage=="policy": policy(config,data,contexts,engine)
    else: test(config,data,contexts,engine)


if __name__=="__main__": main()
