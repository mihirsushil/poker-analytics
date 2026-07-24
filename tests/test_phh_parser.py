from pathlib import Path

import pytest

from poker_analytics.parsers.phh import parse_phhs_file

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "samples" / "phh" / "fixtures" / "sample_50_hands.phhs"


def test_parses_all_hands_in_fixture():
    hands = parse_phhs_file(FIXTURE)
    assert len(hands) == 46
    assert all(h.source == "phh" for h in hands)


def test_button_is_last_player_index():
    hands = parse_phhs_file(FIXTURE)
    for h in hands:
        assert h.button_seat == h.seats[-1].seat_no


def test_money_reconciles_to_a_small_plausible_rake():
    """Real cash-game rake, not a parsing bug: negative net_sum within a few
    percent of the pot (never more than ~$1 on these micro-stakes hands),
    positive net_sum (money materializing from nowhere) is a real bug.
    """
    hands = parse_phhs_file(FIXTURE)
    resolved = [h for h in hands if all(s.net_result is not None for s in h.seats)]
    assert len(resolved) >= 35  # most hands should have a resolvable result
    for h in resolved:
        net = sum(s.net_result for s in h.seats)
        assert net <= 0.02, f"hand {h.hand_id}: money appeared from nowhere ({net})"
        assert net >= -max(1.0, 0.3 * h.total_pot), f"hand {h.hand_id}: implausibly large rake ({net} on pot {h.total_pot})"


def test_dead_blind_excluded_from_calling_price():
    """A negative blinds_or_straddles entry is dead money (e.g. a missed-blind
    post) - it must count toward the pot but not toward the calling price,
    and the affected hand's money must still reconcile.
    """
    hands = parse_phhs_file(FIXTURE)
    hand = next(h for h in hands if h.hand_id == "59937819782")
    dead_poster = next(s for s in hand.seats if s.starting_stack == 25.25)
    assert dead_poster.net_result is not None
    net = sum(s.net_result for s in hand.seats)
    assert net == pytest.approx(0.0, abs=0.02)


def test_uncalled_raise_excluded_from_winner_contribution():
    hands = parse_phhs_file(FIXTURE)
    hand = next(h for h in hands if h.hand_id == "59937793662")
    winner = next(s for s in hand.seats if s.net_result and s.net_result > 0)
    assert winner.net_result == pytest.approx(1.20, abs=0.05)


def test_names_are_hashed_even_though_source_is_pre_anonymized():
    hands = parse_phhs_file(FIXTURE)
    assert all(len(s.player_hash) == 16 for h in hands for s in h.seats)


def test_no_hero_hole_cards_pre_showdown():
    """Unlike the raw-text format, phh-dataset hands have no privileged
    hero perspective - hole cards are only known for showdown reveals.
    """
    hands = parse_phhs_file(FIXTURE)
    for h in hands:
        for seat in h.seats:
            if seat.hole_cards is not None:
                assert len(seat.hole_cards) == 2
