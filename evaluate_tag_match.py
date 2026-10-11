"""Measures "Find matching images" on a real library, read-only.

    .venv/Scripts/python.exe evaluate_tag_match.py "<library folder or catalog.sqlite>"

For every tag with at least --min-members indexed images (the tag or a
descendant, as tag views count them), a random --holdout share of its members
is hidden (several seeds). The rest are the tag's example images; the hidden
members and every image outside the tag are ranked, as the dialog would rank
them. Reported per method, averaged over tags (seeds averaged first):

* AUC-ROC: chance a hidden member outranks an outside image (0.5 = chance);
* precision@k with k = number hidden: share of hidden members in the top k.

Methods: name only (SigLIP text-image cosine, the old behaviour), examples
only, centroid / top-k parts, the blend (tag_match.combine) at fixed name
weights, and the shipped adaptive weight. It also checks the dialog's default
threshold (the median of the examples' own leave-one-out scores): precision,
recall and images selected at that threshold. Images outside the tag include
untagged ones that may belong in it, so precision is a lower bound.

The catalog is opened with SQLite ``mode=ro``; nothing is written. Tag names are
embedded with the bundled text encoder (tags.embedText) unless --no-names.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

import tag_match
from evaluate_tag_sort import embed_names, load, open_catalog

WEIGHTS = (0.1, 0.2, 0.3, 0.5, 1.0)


def auc(scores, positive):
    """AUC-ROC with tied scores counted as half (Mann-Whitney U)."""
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = positive.sum()
    neg = len(scores) - pos
    return (ranks[positive].sum() - pos * (pos + 1) / 2) / (pos * neg)


def precision_at(scores, positive, k):
    # Stable: equal scores keep candidate order.
    top = np.argsort(-scores, kind='stable')[:k]
    return positive[top].mean()


def example_parts(candidates, examples):
    """(centroid cosine, top-k mean) of each candidate row against examples."""
    center = examples.sum(0)
    center /= np.linalg.norm(center)
    sims = candidates @ examples.T
    k = min(tag_match.TOP_K, examples.shape[0])
    top = -np.sort(-sims, axis=1)[:, :k]
    return candidates @ center, top.mean(1)


def member_parts(examples):
    """Leave-one-out parts for each example against the other examples."""
    total = examples.sum(0)
    others = total[None, :] - examples
    others /= np.linalg.norm(others, axis=1, keepdims=True)
    center = (examples * others).sum(1)
    sims = examples @ examples.T
    np.fill_diagonal(sims, -np.inf)
    k = min(tag_match.TOP_K, examples.shape[0] - 1)
    top = -np.sort(-sims, axis=1)[:, :k]
    return center, top.mean(1)


def evaluate_tag(vectors, members, name, holdout, seeds):
    """Metrics for one tag, averaged over seeds: {method: {metric: value}}."""
    n = len(vectors)
    member_mask = np.zeros(n, bool)
    member_mask[members] = True
    outside = np.flatnonzero(~member_mask)
    w = tag_match.CENTROID_WEIGHT
    totals = {}

    def add(method, key, value):
        totals.setdefault(method, {}).setdefault(key, []).append(float(value))

    def ranked(method, scores, positive, hidden_count):
        scores = np.asarray(scores, dtype=np.float64)
        add(method, 'auc', auc(scores, positive))
        add(method, 'p@k', precision_at(scores, positive, hidden_count))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(members)
        hidden_count = max(1, round(holdout * len(members)))
        hidden, kept = shuffled[:hidden_count], shuffled[hidden_count:]
        cand = np.concatenate([hidden, outside])
        positive = np.zeros(len(cand), bool)
        positive[:hidden_count] = True
        center, top = example_parts(vectors[cand], vectors[kept])
        m_center, m_top = member_parts(vectors[kept])
        example = (w * center + (1 - w) * top).tolist()
        m_example = (w * m_center + (1 - w) * m_top).tolist()
        names = m_names = None
        if name is not None:
            names = (vectors[cand] @ name).tolist()
            m_names = (vectors[kept] @ name).tolist()
            ranked('name only', names, positive, hidden_count)
        ranked('examples only', example, positive, hidden_count)
        ranked('centroid only', center, positive, hidden_count)
        ranked('top-k only', top, positive, hidden_count)
        if names is not None:
            for weight in WEIGHTS:
                fixed = tag_match.combine(example, names, m_example, m_names, weight=weight)
                ranked(f'fixed w={weight:g}', fixed['scores'], positive, hidden_count)
        shipped = tag_match.combine(example, names, m_example, m_names)
        method = 'adaptive (shipped)' if names is not None else 'examples only'
        if names is not None:
            ranked(method, shipped['scores'], positive, hidden_count)
            add(method, 'weight', shipped['name_weight'])
        chosen = np.asarray(shipped['scores']) >= shipped['threshold']
        add('default threshold', 'precision', positive[chosen].mean() if chosen.any() else 0)
        add('default threshold', 'recall', positive[chosen].sum() / hidden_count)
        add('default threshold', 'selected', chosen.sum())
    return {m: {k: float(np.mean(v)) for k, v in keys.items()} for m, keys in totals.items()}


def evaluate(tags, ids, vectors, within, names, min_members=10, holdout=0.3, seeds=range(5)):
    per_tag = {}
    for tag_id, tag in sorted(tags.items(), key=lambda kv: kv[1]['name'].casefold()):
        members = np.array([i for i, asset in enumerate(ids) if tag_id in within.get(asset, ())])
        if len(members) < min_members or len(members) >= len(ids):
            continue
        name = None if names is None else names[tag_id] / np.linalg.norm(names[tag_id])
        per_tag[tag['name']] = {'members': len(members),
                                'methods': evaluate_tag(vectors, members, name, holdout, seeds)}
    methods = {}
    for result in per_tag.values():
        for method, values in result['methods'].items():
            for key, value in values.items():
                methods.setdefault(method, {}).setdefault(key, []).append(value)
    summary = {m: {k: float(np.mean(v)) for k, v in keys.items()} for m, keys in methods.items()}
    return summary, per_tag


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('library', help='library folder or its catalog.sqlite')
    parser.add_argument('--min-members', type=int, default=10)
    parser.add_argument('--holdout', type=float, default=0.3)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--no-names', action='store_true', help='skip the text encoder')
    parser.add_argument('--per-tag', action='store_true', help='print each tag')
    parser.add_argument('--json', help='also write results to this file')
    args = parser.parse_args(argv)
    db = open_catalog(args.library)
    key, tags, ids, vectors, within = load(db)
    db.close()
    names = None if args.no_names else embed_names(tags, key)
    summary, per_tag = evaluate(tags, ids, vectors, within, names, args.min_members,
                                args.holdout, range(args.seeds))
    print(f'{len(per_tag)} tags with >= {args.min_members} members, {len(ids)} indexed images, '
          f'{args.holdout:.0%} hidden, {args.seeds} seeds')
    for method, values in summary.items():
        print(f'  {method:16s} ' + '   '.join(f'{k} {v:.3f}' for k, v in values.items()))
    if args.per_tag:
        for tag, result in per_tag.items():
            row = result['methods']
            line = '  '.join(f'{m}: {row[m]["auc"]:.3f}/{row[m]["p@k"]:.2f}'
                             for m in ('name only', 'examples only', 'adaptive (shipped)')
                             if m in row)
            print(f'  {tag[:24]:24s} {result["members"]:5d}  {line}')
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'spec': tag_match.SPEC_VERSION, 'indexed images': len(ids), 'tags': len(per_tag),
             'summary': summary, 'per tag': per_tag}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
