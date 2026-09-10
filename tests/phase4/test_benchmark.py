"""bucket_prediction() is the one place free-form model text gets mapped to
a gradeable action bucket - it's the part most likely to silently misgrade
a correct prediction as wrong (or vice versa), so it gets the most cases."""
from poker_analytics.phase4.benchmark import bucket_prediction


def test_exact_lowercase_matches():
    assert bucket_prediction("fold") == "fold"
    assert bucket_prediction("check") == "check"
    assert bucket_prediction("call") == "call"


def test_bet_and_raise_with_sizing():
    assert bucket_prediction("bet 25") == "bet"
    assert bucket_prediction("raise to 100") == "raise"


def test_strips_quotes_and_trailing_punctuation():
    assert bucket_prediction('"bet 5000."') == "bet"
    assert bucket_prediction("Fold.") == "fold"


def test_strips_surrounding_whitespace():
    assert bucket_prediction("  Call  ") == "call"


def test_case_insensitive():
    assert bucket_prediction("RAISE to 40") == "raise"


def test_unrecognized_text_is_unparseable():
    assert bucket_prediction("I would need more information to decide.") == "unparseable"
    assert bucket_prediction("") == "unparseable"
