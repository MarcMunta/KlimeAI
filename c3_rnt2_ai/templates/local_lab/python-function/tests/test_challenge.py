from src.challenge import score_attempt


def test_score_attempt_ignores_negatives_and_odds() -> None:
    assert score_attempt([5, 2, -4, 8, 3, 0]) == 10


def test_score_attempt_handles_empty_input() -> None:
    assert score_attempt([]) == 0
