"""Scoring math for AI sort; tiny synthetic vectors, no model weights."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import struct

import pytest

import evaluate_tag_sort
import tag_sort

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'tag_sort_spec.json'
FLUTTER_COPY = Path(__file__).resolve().parents[2] / 'umbra-tags-flutter' / 'test' / 'fixtures' / 'tag_sort_spec.json'


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def test_constants_match_the_shared_fixture():
    fixture = load_fixture()
    assert fixture['spec'] == tag_sort.SPEC_VERSION
    for name, value in fixture['constants'].items():
        assert getattr(tag_sort, name) == value


@pytest.mark.parametrize('case', load_fixture()['cases'], ids=lambda case: case['name'])
def test_reproduces_the_shared_fixture(case):
    rows = tag_sort.score(case['image'], [
        {'name_vector': c['nameVector'], 'members': [(m['id'], m['vector']) for m in c['members']]}
        for c in case['candidates']], case['excludeId'])
    for row, expected in zip(rows, case['expected'], strict=True):
        for key, value in expected.items():
            assert row[key] == pytest.approx(value, abs=1e-9), key


def test_flutter_copy_of_the_fixture_is_identical():
    if not FLUTTER_COPY.is_file():
        pytest.skip('Flutter repository is not checked out beside this one.')
    assert json.loads(FLUTTER_COPY.read_text(encoding='utf-8')) == load_fixture()


def test_softmax_ignores_missing_values():
    assert tag_sort.softmax([None, None], 0.1) == [0.0, 0.0]
    shares = tag_sort.softmax([0.5, None, 0.5], 0.1)
    assert shares == pytest.approx([0.5, 0.0, 0.5])
    # A lower temperature sharpens the same gap.
    assert tag_sort.softmax([0.6, 0.5], 0.01)[0] > tag_sort.softmax([0.6, 0.5], 0.1)[0]


def test_examples_outweigh_a_misleading_name():
    rows = tag_sort.combine(
        name_cos=[0.05, 0.08],               # the name favours candidate 1 ...
        centroid_cos=[0.9, 0.5],             # ... the examples favour candidate 0
        neighbours=[(0.92, {0}), (0.9, {0}), (0.6, {1})])
    assert rows[0]['confidence'] > rows[1]['confidence']
    assert rows[1]['name'] > rows[0]['name']
    assert sum(r['confidence'] for r in rows) == pytest.approx(1)
    assert [r['votes'] for r in rows] == [2, 1]


def test_without_examples_the_name_decides():
    rows = tag_sort.combine([0.10, 0.12], [None, None], [])
    assert rows[1]['confidence'] > rows[0]['confidence']
    assert rows[0]['confidence'] == rows[0]['name']


def test_no_signal_gives_zero_confidence():
    rows = tag_sort.combine([None, None], [None, None], [])
    assert [r['confidence'] for r in rows] == [0.0, 0.0]


def test_the_image_never_votes_for_itself():
    image = [1.0, 0.0]
    rows = tag_sort.score(image, [
        {'name_vector': None, 'members': [('me', [1.0, 0.0])]},
        {'name_vector': None, 'members': [('other', [0.0, 1.0])]},
    ], exclude_id='me')
    assert rows[0]['members'] == 0 and rows[0]['confidence'] == 0
    assert rows[1]['confidence'] == pytest.approx(1)


def test_rejects_a_zero_image_vector():
    with pytest.raises(ValueError):
        tag_sort.score([0.0, 0.0], [{'name_vector': None, 'members': []}])


# Evaluation script ----------------------------------------------------------

def _catalog(path):
    db = sqlite3.connect(path)
    db.executescript('''
        CREATE TABLE tags(id TEXT PRIMARY KEY, name TEXT, parent_id TEXT, excluded_from_all INTEGER DEFAULT 0);
        CREATE TABLE assets(id TEXT PRIMARY KEY, sha256 TEXT, archived INTEGER DEFAULT 0);
        CREATE TABLE asset_tags(asset_id TEXT, tag_id TEXT);
        CREATE TABLE embeddings(asset_id TEXT, model_key TEXT, analyzed_sha256 TEXT, vector BLOB);''')
    db.executemany('INSERT INTO tags(id,name,parent_id) VALUES (?,?,?)', [
        ('red', 'red', None), ('blue', 'blue', None), ('dark', 'dark', 'red'), ('light', 'light', 'red')])
    groups = {'dark': (1, 0.1, 0), 'light': (1, 0, 0.2), 'blue': (0, 1, 0.1)}
    for tag, base in groups.items():
        for i in range(4):
            asset = f'{tag}{i}'
            vector = [b + 0.01 * i * (k + 1) for k, b in enumerate(base)]
            norm = math.sqrt(sum(v * v for v in vector))
            db.execute('INSERT INTO assets(id,sha256) VALUES (?,?)', [asset, asset])
            db.execute('INSERT INTO asset_tags VALUES (?,?)', [asset, tag])
            db.execute('INSERT INTO embeddings VALUES (?,?,?,?)',
                       [asset, 'k', asset, struct.pack('<3f', *[v / norm for v in vector])])
    db.commit()
    db.close()


def test_evaluation_runs_read_only_on_a_tiny_catalog(tmp_path, capsys):
    library = tmp_path / 'library'
    library.mkdir()
    catalog = library / 'catalog.sqlite'
    _catalog(catalog)
    before = sha256(catalog.read_bytes()).hexdigest()
    out = tmp_path / 'results.json'
    evaluate_tag_sort.main([str(library), '--no-names', '--json', str(out)])
    assert sha256(catalog.read_bytes()).hexdigest() == before
    results = json.loads(out.read_text(encoding='utf-8'))
    assert results['mode 1']['images'] == 12 and results['mode 1']['candidates'] == 2
    assert results['mode 1']['methods']['blended']['top1'] == 1.0
    assert results['mode 2']['images'] == 8
    assert 'name only' not in results['mode 1']['methods']
    assert 'Mode 2 - children of red' in capsys.readouterr().out
