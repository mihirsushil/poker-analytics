from pathlib import Path

import pytest

from poker_analytics.parsers.pokerstars_text import parse_file

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "samples" / "raw_text"
ALL_FIXTURES = sorted(FIXTURES.glob("*.txt"))


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_every_fixture_parses_without_error(path):
    hands = parse_file(path)
    assert len(hands) == 1
    hand = hands[0]
    assert hand.hand_id.isdigit()
    assert len(hand.seats) >= 2
    assert hand.button_seat in {s.seat_no for s in hand.seats}
    # Every seat must get a position label.
    assert all(s.position for s in hand.seats)


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_money_reconciles_within_rake(path):
    """sum(contributed) - sum(collected) should equal rake (chip conservation)."""
    hand = parse_file(path)[0]
    net_sum = sum(s.net_result for s in hand.seats)
    assert net_sum == pytest.approx(-hand.rake, abs=0.02)


def test_all_in_flag_captured_on_short_stack_partial_blind():
    hand = parse_file(FIXTURES / "posts-allin.txt")[0]
    all_in_actions = [a for a in hand.actions if a.is_all_in]
    assert len(all_in_actions) >= 1


def test_uncalled_bet_reduces_contribution_not_result():
    hand = parse_file(FIXTURES / "uncalled-bet-returned.txt")[0]
    # GuiTrettel and held split the pot 2030/2030 after a chunk of GuiTrettel's
    # all-in bet (325) was returned uncalled - both should show a real net
    # result, not one inflated by the returned chips.
    winners = [s for s in hand.seats if s.net_result and s.net_result > 0]
    assert len(winners) == 2


def test_side_pot_split_correctly():
    hand = parse_file(FIXTURES / "posts-allin.txt")[0]
    assert len(hand.pots) == 2
    amounts = sorted(p.amount for p in hand.pots)
    assert amounts == [67, 938]
    assert hand.total_pot == 1005


def test_sitting_out_player_still_gets_a_seat_and_position():
    hand = parse_file(FIXTURES / "player-sitting-out.txt")[0]
    sitting_out_seats = [s for s in hand.seats if s.sitting_out]
    assert len(sitting_out_seats) == 1
    assert sitting_out_seats[0].position is not None


def test_legacy_game_header_format_parses():
    hand = parse_file(FIXTURES / "2008-tournament.txt")[0]
    assert hand.is_tournament
    assert hand.played_at.year == 2008


def test_cash_game_currency_detected():
    hand = parse_file(FIXTURES / "cash.zoom.2017.txt")[0]
    assert hand.currency == "$"
    assert not hand.is_tournament


def test_names_with_parens_and_spaces_parsed_as_single_player():
    hand = parse_file(FIXTURES / "cash.zoom.2017.parens-in-playername.txt")[0]
    # 6 seats, 6 distinct player hashes, despite screen names containing "(" and ")"
    assert len(hand.seats) == 6
    assert len({s.player_hash for s in hand.seats}) == 6


def test_hero_hole_cards_captured_and_names_hashed():
    hand = parse_file(FIXTURES / "cash.zoom.2017.txt")[0]
    hero = hand.hero_seat()
    assert hero is not None
    assert len(hero.hole_cards) == 2
    # No raw screen name should leak into any player_hash.
    assert all(len(s.player_hash) == 16 for s in hand.seats)
