from text import TextProblemsEmulator


def test_init():
    text_problems_emulator = TextProblemsEmulator()
    assert text_problems_emulator, "could not init TextProblemsEmulator"
