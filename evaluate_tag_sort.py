"""Measures "AI sort into existing tags" on a real library, read-only.

    .venv/Scripts/python.exe evaluate_tag_sort.py "<library folder or catalog.sqlite>"

Leave-one-out over images that are already tagged: each image is scored as if
it were unsorted, against the other images' tags, for

* mode 1, top-level tags: correct when the best suggestion is one of the
  image's top-level tags (the roots above its tags);
* mode 2, child tags: for every tag with at least two children, images in one
  of its children; correct when the best suggestion is one of those children.

Methods: name only (SigLIP text-image), examples only (kNN vote + centroid),
and the blend the app uses (tag_sort.combine). ``--shots N`` keeps at most N
random example images per tag (the rest are scored), simulating new tags.

The catalog is opened with SQLite ``mode=ro``; nothing is written. Tag names are
embedded with the bundled text encoder (tags.embedText) unless --no-names.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

import tag_sort

ROOT = Path(__file__).resolve().parent


def open_catalog(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'catalog.sqlite'
    if not path.is_file():
        raise SystemExit(f'No catalog at {path}')
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def load(db, key=None):
    tags = {row[0]: {'name': row[1], 'parent': row[2], 'reference': bool(row[3])}
            for row in db.execute('SELECT id,name,parent_id,excluded_from_all FROM tags')}
    if key is None:
        row = db.execute('''SELECT model_key FROM embeddings WHERE vector IS NOT NULL
            GROUP BY model_key ORDER BY COUNT(*) DESC LIMIT 1''').fetchone()
        if row is None:
            raise SystemExit('The library has no image embeddings yet.')
        key = row[0]
    rows = db.execute('''SELECT a.id,e.vector FROM assets a JOIN embeddings e ON e.asset_id=a.id
        WHERE e.model_key=? AND e.analyzed_sha256=a.sha256 AND e.vector IS NOT NULL
        AND a.archived=0 ORDER BY a.id''', [key]).fetchall()
    ids = [row[0] for row in rows]
    vectors = np.stack([np.frombuffer(row[1], dtype='<f4') for row in rows]).astype(np.float64)
    index = {asset: i for i, asset in enumerate(ids)}
    direct = defaultdict(set)
    for asset, tag in db.execute('SELECT asset_id,tag_id FROM asset_tags'):
        if asset in index and tag in tags:
            direct[asset].add(tag)
    # Each image's tags with all their ancestors: membership of every subtree.
    lineage = {}
    for tag in tags:
        chain, at = [], tag
        while at is not None and at in tags and at not in chain:
            chain.append(at)
            at = tags[at]['parent']
        lineage[tag] = chain
    within = {asset: {a for tag in labels for a in lineage[tag]} for asset, labels in direct.items()}
    return key, tags, ids, vectors, within


def embed_names(tags, key):
    from tag_worker import TagWorker
    worker = TagWorker(ROOT)
    vectors = {}
    for tag_id, tag in tags.items():
        result = worker.dispatch({'method': 'tags.embedText', 'text': tag['name']})
        if result.get('compatibleEmbeddingKey') != key:
            raise SystemExit('The bundled text encoder does not match the library embeddings.')
        vectors[tag_id] = np.asarray(result['vector'], dtype=np.float64)
    return vectors


def evaluate(candidates, ids, vectors, within, names, shots=None, seed=0):
    """Returns {method: [top1 hits, top3 hits]} and the number of images scored."""
    members = {c: [i for i, asset in enumerate(ids) if c in within.get(asset, ())] for c in candidates}
    samples = [(i, {j for j, c in enumerate(candidates) if c in within.get(asset, ())})
               for i, asset in enumerate(ids)]
    samples = [(i, truth) for i, truth in samples if truth]
    if shots:
        rng = np.random.default_rng(seed)
        for c in candidates:
            if len(members[c]) > shots:
                members[c] = sorted(rng.choice(members[c], shots, replace=False).tolist())
        kept = {i for m in members.values() for i in m}
        samples = [(i, truth) for i, truth in samples if i not in kept]
    member_sets = [set(members[c]) for c in candidates]
    sums = [vectors[members[c]].sum(0) if members[c] else np.zeros(vectors.shape[1]) for c in candidates]
    pool = sorted(set().union(*member_sets))
    pool_vectors = vectors[pool]
    pool_labels = [{j for j, s in enumerate(member_sets) if p in s} for p in pool]
    pool_ids = np.array([ids[p] for p in pool])
    text = None if names is None else np.stack([names[c] for c in candidates])
    methods = {'name only': None, 'examples only': None, 'kNN only': None,
               'centroid only': None, 'blended': None}
    hits = {m: [0, 0] for m in methods}
    for i, truth in samples:
        x = vectors[i]
        name_cos = [None] * len(candidates) if text is None else list(text @ x)
        centroid_cos = []
        for j, c in enumerate(candidates):
            total = sums[j] - (x if i in member_sets[j] else 0)
            count = len(members[c]) - (i in member_sets[j])
            norm = np.linalg.norm(total)
            centroid_cos.append(float(x @ total / norm) if count and norm > 0 else None)
        sims = pool_vectors @ x
        order = [p for p in np.lexsort((pool_ids, -sims)) if pool[p] != i][:tag_sort.NEIGHBOURS]
        neighbours = [(float(sims[p]), pool_labels[p]) for p in order]
        blended = tag_sort.combine(name_cos, centroid_cos, neighbours)
        examples = tag_sort.combine([None] * len(candidates), centroid_cos, neighbours)
        scores = {
            'blended': [r['confidence'] for r in blended],
            'examples only': [r['confidence'] for r in examples],
            'kNN only': [r['knn'] for r in examples],
            'centroid only': [r['centroid'] for r in examples],
        }
        if text is not None:
            scores['name only'] = [r['name'] for r in blended]
        for method, values in scores.items():
            # Stable: ties keep candidate order, as in the app.
            ranked = sorted(range(len(candidates)), key=lambda j: -values[j])
            hits[method][0] += ranked[0] in truth
            hits[method][1] += bool(set(ranked[:3]) & truth)
    return {m: h for m, h in hits.items() if text is not None or m != 'name only'}, len(samples)


def report(title, hits, total, candidates):
    print(f'\n{title}: {total} images, {candidates} candidate tags')
    for method, (top1, top3) in hits.items():
        if total:
            print(f'  {method:14s} top-1 {top1 / total:6.1%}   top-3 {top3 / total:6.1%}')
    return {'images': total, 'candidates': candidates,
            'methods': {m: {'top1': h[0] / total, 'top3': h[1] / total} for m, h in hits.items()} if total else {}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('library', help='library folder or its catalog.sqlite')
    parser.add_argument('--shots', type=int, help='keep at most N example images per tag')
    parser.add_argument('--no-names', action='store_true', help='skip the text encoder')
    parser.add_argument('--json', help='also write results to this file')
    args = parser.parse_args(argv)
    db = open_catalog(args.library)
    key, tags, ids, vectors, within = load(db)
    db.close()
    names = None if args.no_names else embed_names(tags, key)
    results = {'spec': tag_sort.SPEC_VERSION, 'indexed images': len(ids), 'shots': args.shots}
    top = [t for t, v in sorted(tags.items(), key=lambda kv: kv[1]['name'].casefold()) if v['parent'] is None]
    hits, total = evaluate(top, ids, vectors, within, names, args.shots)
    results['mode 1'] = report('Mode 1 - top-level tags', hits, total, len(top))
    children = defaultdict(list)
    for t, v in sorted(tags.items(), key=lambda kv: kv[1]['name'].casefold()):
        if v['parent'] is not None:
            children[v['parent']].append(t)
    combined, total2, parents = defaultdict(lambda: [0, 0]), 0, []
    for parent, kids in children.items():
        if len(kids) < 2:
            continue
        hits, total = evaluate(kids, ids, vectors, within, names, args.shots)
        parents.append(report(f'Mode 2 - children of {tags[parent]["name"]}', hits, total, len(kids)))
        total2 += total
        for m, (a, b) in hits.items():
            combined[m][0] += a
            combined[m][1] += b
    results['mode 2 parents'] = parents
    title = f'Mode 2 - all {len(parents)} parents with 2+ children'
    if len(parents) == 1:
        results['mode 2'] = parents[0]
    else:
        results['mode 2'] = report(title, dict(combined), total2,
                                   sum(p['candidates'] for p in parents))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
