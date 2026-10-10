"""Scoring spec for "AI sort into existing tags".

For one image and a set of candidate tags (the top-level tags, or the children
of one tag), rank the candidates using two signals:

* examples: the images already in each candidate's subtree. A soft k-nearest-
  neighbour vote over the labelled images, blended with a softmax over the
  cosine to each candidate's centroid (mean of its unit vectors).
* name: SigLIP 2 text-image cosine between the image and the tag name, encoded
  with the same prompt as ``tags.suggest`` / ``tags.embedText``.

Each signal becomes a distribution over the candidates, so their different
scales (image-image cosine ~0.4-0.95, text-image cosine ~0.0-0.2) never meet
directly. ``evaluate_tag_sort.py`` measured the constants below.

MIRRORED IN DART: umbra-tags-flutter/lib/ml/tag_sort.dart implements the same
formulas. Change both together; ``tests/fixtures/tag_sort_spec.json`` (an
identical copy lives in the Flutter repo's ``test/fixtures``) pins the outputs
and both test suites check against it.
"""
import math

SPEC_VERSION = 1
NEIGHBOURS = 10              # labelled images that vote
NEIGHBOUR_TEMPERATURE = 0.05  # vote weight exp((cos - 1) / T)
CENTROID_TEMPERATURE = 0.03   # softmax over centroid cosines
NAME_TEMPERATURE = 0.01       # softmax over text-image cosines
KNN_WEIGHT = 0.5              # examples = KNN_WEIGHT*knn + (1-KNN_WEIGHT)*centroid
NAME_WEIGHT = 0.2             # confidence = (examples + NAME_WEIGHT*name) / (1+NAME_WEIGHT)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _unit(vector):
    norm = math.sqrt(_dot(vector, vector))
    if not norm > 0 or not math.isfinite(norm):
        raise ValueError('Vector must be finite and non-zero.')
    return [v / norm for v in vector]


def softmax(values, temperature):
    """Softmax over the non-None values; None stays 0. All None gives zeros."""
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    top = max(present)
    weights = [0.0 if v is None else math.exp((v - top) / temperature) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def combine(name_cos, centroid_cos, neighbours):
    """Ranks candidates from precomputed similarities.

    name_cos[j]: text-image cosine for candidate j, or None without a name vector.
    centroid_cos[j]: cosine to candidate j's centroid, or None without members.
    neighbours: the NEIGHBOURS most similar labelled images, as
        (cosine, set of candidate indices the image belongs to).
    Returns one dict per candidate: confidence (all sum to 1 unless there is
    no signal at all), knn, centroid and name shares, votes (neighbours in
    the candidate) and neighbours (how many voted in all).
    """
    count = len(centroid_cos)
    mass = [0.0] * count
    votes = [0] * count
    for cosine, members in neighbours:
        weight = math.exp((cosine - 1) / NEIGHBOUR_TEMPERATURE)
        for j in members:
            mass[j] += weight
            votes[j] += 1
    total = sum(mass)
    knn = [m / total if total > 0 else 0.0 for m in mass]
    centroid = softmax(centroid_cos, CENTROID_TEMPERATURE)
    name = softmax(name_cos, NAME_TEMPERATURE)
    has_examples = any(c is not None for c in centroid_cos)
    has_names = any(n is not None for n in name_cos)
    rows = []
    for j in range(count):
        example = KNN_WEIGHT * knn[j] + (1 - KNN_WEIGHT) * centroid[j]
        if has_examples and has_names:
            confidence = (example + NAME_WEIGHT * name[j]) / (1 + NAME_WEIGHT)
        elif has_examples:
            confidence = example
        else:
            confidence = name[j]
        rows.append({'confidence': confidence, 'example': example, 'knn': knn[j],
                     'centroid': centroid[j], 'name': name[j], 'votes': votes[j],
                     'neighbours': len(neighbours)})
    return rows


def score(image, candidates, exclude_id=None):
    """Reference implementation over plain lists.

    image: the image's vector. candidates: dicts with 'name_vector' (or None)
    and 'members': [(asset_id, unit vector)]. Members equal to exclude_id (the
    image itself) are ignored. Ties among neighbours break by asset id.
    """
    x = _unit(image)
    name_cos, centroid_cos, pool = [], [], {}
    for j, candidate in enumerate(candidates):
        vector = candidate.get('name_vector')
        name_cos.append(None if vector is None else _dot(x, _unit(vector)))
        total, n = None, 0
        for asset_id, member in candidate['members']:
            if asset_id == exclude_id:
                continue
            total = list(member) if total is None else [a + b for a, b in zip(total, member)]
            n += 1
            pool.setdefault(asset_id, (member, set()))[1].add(j)
        norm = math.sqrt(_dot(total, total)) if total else 0.0
        centroid_cos.append(_dot(x, total) / norm if n and norm > 0 else None)
    ranked = sorted(((_dot(x, member), asset_id, labels)
                     for asset_id, (member, labels) in pool.items()),
                    key=lambda row: (-row[0], row[1]))
    rows = combine(name_cos, centroid_cos,
                   [(cosine, labels) for cosine, _, labels in ranked[:NEIGHBOURS]])
    for j, row in enumerate(rows):
        row['members'] = sum(1 for asset_id, _ in candidates[j]['members'] if asset_id != exclude_id)
    return rows
