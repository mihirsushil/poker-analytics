from poker_analytics.phase4.dataset import format_prompt, split, to_chat_example
from poker_analytics.phase4.extract import DecisionExample


def _example(**overrides) -> DecisionExample:
    defaults = dict(
        hand_pk=1, site="PokerStars", source="phh", street="flop", position="BTN",
        hole_cards="Ah,Kh", board_so_far="2c,7d,9h", pot_before=10.0, stack_before=90.0,
        action_history="UTG: fold | BB: check", action_type="bet", amount=5.0, to_amount=None,
        action_label="bet 5",
    )
    defaults.update(overrides)
    return DecisionExample(**defaults)


def test_format_prompt_includes_all_state_fields():
    prompt = format_prompt(_example())
    assert "flop" in prompt
    assert "BTN" in prompt
    assert "Ah,Kh" in prompt
    assert "2c,7d,9h" in prompt
    assert "10.0" in prompt
    assert "90.0" in prompt
    assert "UTG: fold | BB: check" in prompt


def test_format_prompt_labels_empty_board_as_preflop():
    prompt = format_prompt(_example(board_so_far=""))
    assert "no board yet" in prompt


def test_to_chat_example_shape():
    chat = to_chat_example(_example())
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert chat["messages"][-1]["content"] == "bet 5"
    assert chat["action_type"] == "bet"


def test_split_is_deterministic_for_a_fixed_seed():
    examples = [_example(hand_pk=i) for i in range(100)]
    train_a, val_a, test_a = split(examples, seed=42)
    train_b, val_b, test_b = split(examples, seed=42)
    assert [e.hand_pk for e in train_a] == [e.hand_pk for e in train_b]
    assert [e.hand_pk for e in val_a] == [e.hand_pk for e in val_b]
    assert [e.hand_pk for e in test_a] == [e.hand_pk for e in test_b]


def test_split_has_no_overlap_and_covers_everything():
    examples = [_example(hand_pk=i) for i in range(100)]
    train, val, test = split(examples, seed=0, val_frac=0.1, test_frac=0.2)
    train_ids, val_ids, test_ids = (
        {e.hand_pk for e in train}, {e.hand_pk for e in val}, {e.hand_pk for e in test},
    )
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == {e.hand_pk for e in examples}
    assert len(val_ids) == 10
    assert len(test_ids) == 20


def test_split_groups_by_hand_pk_when_multiple_decisions_per_hand():
    # 50 hands, each with 4 decisions (preflop, flop, turn, river) = 200 examples.
    # No hand_pk may appear in more than one split partition.
    examples = [
        _example(hand_pk=hand_id, street=street)
        for hand_id in range(50)
        for street in ("preflop", "flop", "turn", "river")
    ]
    train, val, test = split(examples, seed=7, val_frac=0.1, test_frac=0.2)

    train_hands = {e.hand_pk for e in train}
    val_hands = {e.hand_pk for e in val}
    test_hands = {e.hand_pk for e in test}

    assert not (train_hands & val_hands)
    assert not (train_hands & test_hands)
    assert not (val_hands & test_hands)
    assert len(val_hands) == 5
    assert len(test_hands) == 10
    assert len(train_hands) == 35
    assert len(train) + len(val) + len(test) == 200
