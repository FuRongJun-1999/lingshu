"""Replay of one logical event must not become another independent observation."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world import world_process as WP


def event(tick, eid, kind, **extra):
    return {'tick': tick, 'eid': eid, 'event': kind, **extra}


def development():
    return [event(1, 'a', 'stop'), event(3, 'b', 'turn'),
            event(11, 'a', 'stop'), event(13, 'b', 'turn')]


def holdout():
    return [event(21, 'a', 'stop'), event(23, 'b', 'turn'),
            event(31, 'a', 'stop'), event(33, 'b', 'turn'), event(40, 'c', 'stop')]


def test_duplicate_rows_cannot_manufacture_support():
    one = development()[:2]
    assert WP.echo_lag_associations(one, max_lag=4, min_support=2) == []
    result = WP.echo_lag_associations(one * 3, max_lag=4, min_support=2)
    assert result == [], 'duplicate rows cannot manufacture independent support'


def test_associations_are_invariant_to_replayed_rows():
    unique = development()
    expected = WP.echo_lag_associations(unique, max_lag=4)
    assert expected[0]['count'] == 2 and expected[0]['sources_total'] == 2
    assert WP.echo_lag_associations(unique * 3, max_lag=4) == expected


def test_split_is_unchanged_by_replayed_source_rows():
    unique = development()
    expected = WP.echo_split(unique, 0.3)
    assert WP.echo_split(unique + [unique[0]] * 8, 0.3) == expected
    assert expected['dev_last_tick'] < expected['holdout_first_tick']


def test_evaluation_is_invariant_to_replayed_rows():
    dev, held = development(), holdout()
    expected = WP.evaluate_event_echo(dev, held, max_lag=4)
    assert expected['lag_model']['hits'] == 2
    actual = WP.evaluate_event_echo(dev * 3, held * 3, max_lag=4)
    assert actual == expected, 'replay must not change held-out metrics or model selection'


def test_development_count_uses_same_event_identity_as_learning():
    result = WP.evaluate_event_echo(development() * 3, holdout(), max_lag=4)
    assert result['dev_events'] == 4, 'reported development count must use unique identities'
    assert result['holdout_events'] == 5


def test_distinct_signatures_at_one_tick_are_preserved():
    events = [event(1, 'a', 'stop'), event(1, 'b', 'stop'),
              event(1, 'a', 'turn'), event(2, 'b', 'turn')]
    result = WP.echo_split(events, 0)
    keys = {(e['tick'], e['eid'], e['event']) for e in result['dev']}
    assert len(keys) == 4, 'distinct eid/event at one tick must survive'
    assert result['events'] == result['dev_events'] == 4


def test_input_is_read_only_and_duplicate_keeps_first_payload():
    original = development()
    original[0]['dt'] = 1
    replay = copy.deepcopy(original[0])
    replay['dt'] = 999
    rows = original + [replay]
    saved = copy.deepcopy(rows)
    result = WP.echo_split(rows, 0)
    assert result['events'] == 4
    assert result['dev'][0]['dt'] == 1
    assert rows == saved and result == WP.echo_split(rows, 0)
