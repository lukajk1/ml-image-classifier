"""Scoring math for "Find matching images"; tiny synthetic vectors, no model weights."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

import evaluate_tag_match
import tag_match
from test_tag_sort import _catalog

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'tag_match_spec.json'
FLUTTER_COPY = Path(__file__).resolve().parents[2] / 'umbra-tags-flutter' / 'test' / 'fixtures' / 'tag_match_spec.json'


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def test_constants_match_the_shared_fixture():
    fixture = load_fixture()
    assert fixture['spec'] == tag_match.SPEC_VERSION
    for name, value in fixture['constants'].items():
        assert getattr(tag_match, name) == value


@pytest.mark.parametrize('case', load_fixture()['cases'], ids=lambda case: case['name'])
def test_reproduces_the_shared_fixture(case):
    result = tag_match.score(case['images'], case['members'], case['nameVector'])
    expected = case['expected']
    for key, field in [('scores', 'scores'), ('memberScores', 'member_scores'),
                       ('examples', 'examples'), ('names', 'names')]:
        if expected[key] is None:
            assert result[field] is None, key
        else:
            assert result[field] == pytest.approx(expected[key], abs=1e-9), key
    for key, field in [('nameWeight', 'name_weight'), ('threshold', 'threshold')]:
        assert result[field] == (None if expected[key] is None else pytest.approx(expected[key], abs=1e-9))


def test_flutter_copy_of_the_fixture_is_identical():
    if not FLUTTER_COPY.is_file():
        pytest.skip('Flutter repository is not checked out beside this one.')
    assert json.loads(FLUTTER_COPY.read_text(encoding='utf-8')) == load_fixture()


def test_example_score_blends_centroid_and_nearest_members():
    members = [[1.0, 0.0], [0.0, 1.0]]
    scores, loo = tag_match.example_scores([[1.0, 0.0]], members)
    # Centroid (0.707, 0.707): cos 0.707; top-2 mean of (1, 0) = 0.5.
    assert scores[0] == pytest.approx(0.5 * 2 ** -0.5 + 0.5 * 0.5)
    # Leave-one-out: each member sees only the other, orthogonal one.
    assert loo == pytest.approx([0.0, 0.0])


def test_a_name_that_fits_the_members_gets_weight_and_a_misleading_one_none():
    examples = [0.9, 0.5, 0.4, 0.3]
    names = [0.10, 0.05, 0.02, 0.01]
    member_examples = [0.8, 0.7]
    helpful = tag_match.combine(examples, names, member_examples, [0.12, 0.09])
    misleading = tag_match.combine(examples, names, member_examples, [0.0, 0.01])
    assert 0 < helpful['name_weight'] <= tag_match.MAX_NAME_WEIGHT
    assert misleading['name_weight'] == 0
    assert misleading['scores'] == examples
    # The name only reorders around its mean; example units are kept.
    assert np.mean(helpful['scores']) == pytest.approx(np.mean(examples))


def test_members_no_closer_than_outsiders_give_the_name_full_weight():
    result = tag_match.combine([0.9, 0.5], [0.1, 0.0], [0.5, 0.6], [0.2, 0.2])
    assert result['name_weight'] == tag_match.MAX_NAME_WEIGHT


def test_name_only_and_examples_only():
    assert tag_match.combine(None, [0.1, 0.2])['scores'] == [0.1, 0.2]
    only = tag_match.combine([0.5, 0.6], None, [0.7, 0.8])
    assert only['scores'] == [0.5, 0.6] and only['name_weight'] is None
    assert only['threshold'] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        tag_match.combine(None, None)


def test_quantile_matches_numpy():
    values = [0.3, 0.9, 0.1, 0.5, 0.7]
    for q in (0, 0.2, 0.5, 0.9, 1):
        assert tag_match.quantile(values, q) == pytest.approx(np.quantile(values, q))


def test_vectorized_evaluation_parts_match_the_reference():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(12, 6))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    images, members = vectors[:5], vectors[5:]
    reference, loo = tag_match.example_scores(images.tolist(), members.tolist())
    w = tag_match.CENTROID_WEIGHT
    center, top = evaluate_tag_match.example_parts(images, members)
    assert (w * center + (1 - w) * top).tolist() == pytest.approx(reference)
    m_center, m_top = evaluate_tag_match.member_parts(members)
    assert (w * m_center + (1 - w) * m_top).tolist() == pytest.approx(loo)


def test_auc_counts_ties_as_half():
    scores = np.array([0.9, 0.5, 0.5, 0.1])
    assert evaluate_tag_match.auc(scores, np.array([True, True, False, False])) == pytest.approx(0.875)


def test_evaluation_runs_read_only_on_a_tiny_catalog(tmp_path, capsys):
    library = tmp_path / 'library'
    library.mkdir()
    catalog = library / 'catalog.sqlite'
    _catalog(catalog)
    before = sha256(catalog.read_bytes()).hexdigest()
    out = tmp_path / 'results.json'
    evaluate_tag_match.main([str(library), '--no-names', '--min-members', '4', '--json', str(out)])
    assert sha256(catalog.read_bytes()).hexdigest() == before
    results = json.loads(out.read_text(encoding='utf-8'))
    # red (8, with its children), blue, dark and light (4 each).
    assert results['tags'] == 4
    assert results['summary']['examples only']['auc'] > 0.9
    assert 'name only' not in results['summary']
    assert '4 tags with >= 4 members' in capsys.readouterr().out
