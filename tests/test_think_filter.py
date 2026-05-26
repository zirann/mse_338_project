from redteam.generator import strip_think


def test_strip_think_removes_closed_block() -> None:
    assert strip_think("<think>abc</think>\nFINAL") == "FINAL"


def test_strip_think_rejects_unclosed_prefix() -> None:
    assert strip_think("<think>\nabc") == ""

