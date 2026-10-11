"""Scoring spec for "Find matching images" (one tag, many images).

For a tag, every image outside it is scored from two signals:

* examples: the images already in the tag or one of its descendants
  ("members"). ``example = CENTROID_WEIGHT * cos(x, centroid)
  + (1 - CENTROID_WEIGHT) * mean of the TOP_K highest cos(x, member)``. The
  centroid is the renormalized mean of the members' unit vectors; TOP_K is
  capped by the member count.
* name: SigLIP 2 text-image cosine between the image and the tag name
  (``tags.embedText``, same prompt as ``tags.suggest``).

The two live on different scales (image-image cosine ~0.3-0.95, text-image
~0.0-0.2), so the name is rescaled to the examples' spread over the scored
images:

    score = example + w * (sd(example) / sd(name)) * (name - mean(name))

which ranks like z(example) + w * z(name) but keeps the number in example
(cosine) units: the name adds or subtracts a bonus around zero. Means and
population sds are over the scored images.

w is chosen per tag from the members themselves, like a Fisher discriminant
for two standardized signals: w = d_name / d_example, clamped to
[0, MAX_NAME_WEIGHT], where d = (mean over members - mean over scored images)
/ sd over scored images, and the members' example scores are leave-one-out
(each member against the others). A name that does not describe the tag's
images (d_name <= 0) gets no weight. ``evaluate_tag_match.py`` showed this
beats every fixed weight on a real library (README "Evaluation").

Edge cases: no members -> the raw name cosine; no name vector -> the example
score; one member -> w = SINGLE_MEMBER_NAME_WEIGHT (no leave-one-out
possible); a zero sd -> w = 0.

The dialog's default threshold is the THRESHOLD_QUANTILE quantile (linear
interpolation) of the members' leave-one-out scores: images that match at
least as well as a typical image already in the tag. None below 2 members.

MIRRORED IN DART: umbra-tags-flutter/lib/ml/tag_match.dart implements the same
formulas. Change both together; ``tests/fixtures/tag_match_spec.json`` (an
identical copy lives in the Flutter repo's ``test/fixtures``) pins the outputs
and both test suites check against it.
"""
import math

SPEC_VERSION = 1
TOP_K = 5                       # nearest members averaged
CENTROID_WEIGHT = 0.5           # example = w*centroid + (1-w)*top-k mean
MAX_NAME_WEIGHT = 2.0           # cap on the adaptive name weight
SINGLE_MEMBER_NAME_WEIGHT = 1.0  # name weight when only one member exists
THRESHOLD_QUANTILE = 0.5        # default threshold: median member score


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _unit(vector):
    norm = math.sqrt(_dot(vector, vector))
    if not norm > 0 or not math.isfinite(norm):
        raise ValueError('Vector must be finite and non-zero.')
    return [v / norm for v in vector]


def _example(x, units, total, skip=None):
    """Example score of unit vector x against units (minus index skip)."""
    others = [m for j, m in enumerate(units) if j != skip]
    if not others:
        return None
    center = total if skip is None else [t - v for t, v in zip(total, units[skip])]
    norm = math.sqrt(_dot(center, center))
    center_cos = _dot(x, center) / norm if norm > 0 else 0.0
    cosines = sorted((_dot(x, m) for m in others), reverse=True)
    k = min(TOP_K, len(cosines))
    return CENTROID_WEIGHT * center_cos + (1 - CENTROID_WEIGHT) * sum(cosines[:k]) / k


def example_scores(images, members):
    """Example score of each image, and each member's leave-one-out score
    (None with fewer than 2 members)."""
    units = [_unit(m) for m in members]
    total = [sum(column) for column in zip(*units)]
    scores = [_example(_unit(x), units, total) for x in images]
    loo = [_example(m, units, total, j) for j, m in enumerate(units)] if len(units) >= 2 else None
    return scores, loo


def _stats(values):
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def name_weight(examples, names, member_examples, member_names):
    """The adaptive name weight (see the module docstring)."""
    if len(examples) < 2:
        return 0.0
    mean_e, sd_e = _stats(examples)
    mean_n, sd_n = _stats(names)
    if not sd_e > 0 or not sd_n > 0:
        return 0.0
    if member_examples is None:
        return SINGLE_MEMBER_NAME_WEIGHT
    d_e = (sum(member_examples) / len(member_examples) - mean_e) / sd_e
    d_n = (sum(member_names) / len(member_names) - mean_n) / sd_n
    if d_n <= 0:
        return 0.0
    if d_e <= 0:
        return MAX_NAME_WEIGHT
    return min(MAX_NAME_WEIGHT, d_n / d_e)


def combine(examples, names, member_examples=None, member_names=None, weight=None):
    """Final scores. examples / names are per scored image (either may be
    None: no members / no name vector); member_* are the members' leave-one-
    out example scores and name cosines. weight overrides the adaptive name
    weight (evaluation only). Returns {'scores', 'member_scores',
    'name_weight', 'threshold'}."""
    if examples is None and names is None:
        raise ValueError('Need members or a name vector.')
    if examples is None:
        return {'scores': list(names), 'member_scores': None, 'name_weight': None, 'threshold': None}
    w, scale, mean_n = None, 0.0, 0.0
    if names is not None:
        w = name_weight(examples, names, member_examples, member_names) if weight is None else weight
        if w > 0 and len(examples) >= 2:
            _, sd_e = _stats(examples)
            mean_n, sd_n = _stats(names)
            scale = w * sd_e / sd_n if sd_n > 0 else 0.0
    scores = [e + scale * (n - mean_n) for e, n in zip(examples, names or [0.0] * len(examples))]
    member_scores = None
    if member_examples is not None:
        member_scores = [e + scale * (n - mean_n)
                         for e, n in zip(member_examples, member_names or [0.0] * len(member_examples))]
    return {'scores': scores, 'member_scores': member_scores, 'name_weight': w,
            'threshold': quantile(member_scores, THRESHOLD_QUANTILE) if member_scores else None}


def quantile(values, q):
    """Linear-interpolation quantile (numpy's default method)."""
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    low = math.floor(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (at - low) * (ordered[high] - ordered[low])


def score(images, members, name_vector=None):
    """Reference implementation over plain lists: images are the vectors to
    score, members the tag's image vectors, name_vector the tag name's text
    vector or None. Returns combine()'s dict plus 'examples' and 'names'."""
    examples = loo = None
    if members:
        examples, loo = example_scores(images, members)
    names = member_names = None
    if name_vector is not None:
        text = _unit(name_vector)
        names = [_dot(_unit(x), text) for x in images]
        member_names = [_dot(_unit(m), text) for m in members]
    result = combine(examples, names, loo, member_names)
    result.update(examples=examples, names=names)
    return result
