"""Parse Metamath frames and actual compressed-proof references.

The optional proof checker checks substitutions, essential hypotheses, distinct
variables and final stack expressions. It is a research implementation, not an
independently certified replacement for the upstream verifiers.
"""
from __future__ import annotations
import argparse
import bisect
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import re
import sys
import time


@dataclass(slots=True)
class Statement:
    label: str
    kind: str
    expr: tuple[str, ...]
    hyps: tuple[int, ...]
    distinct: tuple[tuple[str, str], ...]
    deps: tuple[int, ...]
    line: int
    offset: int
    chapter: str
    section: str
    comment: str


def compressed_steps(code: str):
    number = 0
    for char in code:
        if "A" <= char <= "T":
            yield number * 20 + ord(char) - ord("A") + 1
            number = 0
        elif "U" <= char <= "Y":
            number = number * 5 + ord(char) - ord("U") + 1
        elif char == "Z":
            if number:
                raise ValueError("Z within an encoded number")
            yield 0
        elif char == "?":
            raise ValueError("Incomplete proof")
        elif not char.isspace():
            raise ValueError(f"Illegal compressed-proof character: {char!r}")
    if number:
        raise ValueError("Unterminated compressed-proof number")


def substitute(expr, substitutions):
    return tuple(itertools.chain.from_iterable(substitutions.get(x, (x,)) for x in expr))


def execute_proof(target, proof, statements, labels, active, variables, active_distinct, verify):
    compressed = bool(proof and proof[0] == "(")
    if compressed:
        close = proof.index(")")
        reference = list(target.hyps) + [labels[x] for x in proof[1:close]]
        steps = compressed_steps("".join(proof[close + 1:]))
    else:
        reference = [labels[x] for x in proof]
        steps = iter(range(1, len(reference) + 1))
    for index in reference:
        if statements[index].kind in ("$f", "$e") and index not in active:
            raise ValueError(f"Inactive hypothesis in proof of {target.label}")
    dependencies = set()
    stack = []
    saved = []
    saved_count = 0
    for number in steps:
        if number == 0:
            saved_count += 1
            if verify:
                if not stack:
                    raise ValueError("Save on an empty proof stack")
                saved.append(stack[-1])
            continue
        if number > len(reference):
            if number - len(reference) > saved_count:
                raise ValueError("Reference to an unavailable saved step")
            if verify:
                stack.append(saved[number - len(reference) - 1])
            continue
        index = reference[number - 1]
        used = statements[index]
        if used.kind in ("$a", "$p"):
            dependencies.add(index)
        if not verify:
            continue
        if used.kind in ("$f", "$e"):
            stack.append(used.expr)
            continue
        count = len(used.hyps)
        if len(stack) < count:
            raise ValueError(f"Stack underflow applying {used.label} in {target.label}")
        arguments = stack[-count:] if count else []
        if count:
            del stack[-count:]
        substitutions = {}
        for hypothesis, argument in zip(used.hyps, arguments):
            hyp = statements[hypothesis]
            if hyp.kind == "$f":
                if not argument or argument[0] != hyp.expr[0]:
                    raise ValueError(f"Type mismatch in {target.label}: {used.label}")
                substitutions[hyp.expr[1]] = argument[1:]
        for hypothesis, argument in zip(used.hyps, arguments):
            hyp = statements[hypothesis]
            if hyp.kind == "$e" and substitute(hyp.expr, substitutions) != argument:
                raise ValueError(f"Essential-hypothesis mismatch: {target.label}/{used.label}")
        if used.distinct:
            subvars = {v: set(expr) & variables for v, expr in substitutions.items()}
            for x, y in used.distinct:
                for a in subvars[x]:
                    for b in subvars[y]:
                        if a == b or tuple(sorted((a, b))) not in active_distinct:
                            raise ValueError(f"Distinct-variable violation: {target.label}/{used.label}")
        stack.append(substitute(used.expr, substitutions))
    if verify and stack != [target.expr]:
        raise ValueError(f"Final proof expression mismatch: {target.label}")
    return tuple(sorted(dependencies))


def parse_text(text: str, verify=False):
    tokens = iter(re.finditer(r"\$\([\s\S]*?\$\)|\S+", text))
    lines = [m.start() for m in re.finditer("\n", text)]
    statements = []
    labels = {}
    active_hyps = []
    active_distinct = set()
    variables = set()
    scopes = []
    chapter = section = "Preamble"
    comment = ""

    def next_token():
        nonlocal chapter, section, comment
        while True:
            match = next(tokens)
            token = match.group()
            if not token.startswith("$("):
                return token, match.start()
            content = token[2:-2].strip()
            major = re.findall(r"(?m)^[#*]{20,}\s*\n\s*([^\n]+)\n[#*]{20,}", content)
            minor = re.findall(r"(?m)^[=\-]{20,}\s*\n\s*([^\n]+)\n[=\-]{20,}", content)
            if major:
                chapter = major[-1].strip()
                section = chapter
            if minor:
                section = minor[-1].strip()
            comment = content

    def until(end):
        result = []
        while True:
            value, _ = next_token()
            if value == end:
                return result
            result.append(sys.intern(value))

    while True:
        try:
            token, offset = next_token()
        except StopIteration:
            break
        if token == "${":
            scopes.append((len(active_hyps), active_distinct.copy()))
            continue
        if token == "$}":
            size, active_distinct = scopes.pop()
            del active_hyps[size:]
            continue
        if token == "$[":
            raise ValueError("Includes must be expanded before parsing")
        if token in ("$c", "$v", "$d"):
            values = until("$.")
            if token == "$v":
                variables.update(values)
            if token == "$d":
                active_distinct.update(tuple(sorted(x)) for x in itertools.combinations(values, 2))
            continue
        label = token
        kind, _ = next_token()
        if kind not in ("$f", "$e", "$a", "$p"):
            raise ValueError(f"Unknown statement: {label} {kind}")
        body = until("$.")
        proof = []
        if kind == "$p":
            equals = body.index("$=")
            proof = body[equals + 1:]
            body = body[:equals]
        expr = tuple(body)
        mandatory = set(expr) & variables
        for i in active_hyps:
            if statements[i].kind == "$e":
                mandatory.update(set(statements[i].expr) & variables)
        hyps = tuple(i for i in active_hyps if statements[i].kind == "$e" or statements[i].expr[1] in mandatory)
        distinct = tuple(sorted((a, b) for a, b in active_distinct if a in mandatory and b in mandatory))
        stmt = Statement(label, kind, expr, hyps if kind in ("$a", "$p") else (),
                         distinct if kind in ("$a", "$p") else (), (),
                         bisect.bisect_left(lines, offset) + 1, offset, chapter, section, comment)
        if kind == "$p":
            stmt.deps = execute_proof(stmt, proof, statements, labels, set(active_hyps),
                                      variables, active_distinct, verify)
        if label in labels:
            raise ValueError(f"Duplicate label {label}")
        labels[label] = len(statements)
        if kind in ("$f", "$e"):
            active_hyps.append(len(statements))
        statements.append(stmt)
        comment = ""
    if scopes:
        raise ValueError("Unclosed scope")
    return statements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["set", "iset", "hol"], default="set")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = Path("data/raw") / (args.corpus + ".mm")
    t = time.perf_counter()
    raw = path.read_bytes()
    statements = parse_text(raw.decode("utf-8"), verify=args.verify)
    assertions = [i for i, s in enumerate(statements) if s.kind in ("$a", "$p") and s.expr[0] == "|-"]
    aset = set(assertions)
    edges = [(i, j) for i in assertions for j in statements[i].deps if j in aset]
    stats = {"corpus": args.corpus, "source_sha256": hashlib.sha256(raw).hexdigest(),
             "all_statements": len(statements), "statement_kinds": dict(Counter(s.kind for s in statements)),
             "assertion_nodes": len(assertions), "proof_uses_edges": len(edges),
             "has_essential_hypothesis_edges": sum(sum(statements[h].kind == "$e" for h in statements[i].hyps) for i in assertions),
             "chapters": dict(Counter(statements[i].chapter for i in assertions)),
             "sections": len(set((statements[i].chapter, statements[i].section) for i in assertions)),
             "proofs_checked_by_local_checker": sum(s.kind == "$p" for s in statements) if args.verify else 0,
             "seconds": time.perf_counter() - t}
    root = Path("data/processed")
    root.mkdir(parents=True, exist_ok=True)
    with (root / (args.corpus + ".pkl")).open("wb") as f:
        pickle.dump({"statements": statements, "assertions": assertions, "edges": edges, "stats": stats}, f, protocol=5)
    (root / (args.corpus + "_stats.json")).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
