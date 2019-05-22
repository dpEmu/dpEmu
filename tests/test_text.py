'''
from mock import patch

from text import TextProblemsEmulator


@patch("random.choices", return_value=["q"])
def test_replace_chars_given_probs_with_replacement(choices_mock):
    text_problems_emulator = TextProblemsEmulator()
    text_problems_emulator.text = "ggqq99"
    replacements = {
        "g": [["q", "9"], [.5, .3]],
    }
    text_problems_emulator.replace_chars_given_probs(replacements)
    assert text_problems_emulator.text == "qqqq99"


@patch("random.choices", return_value=["g"])
def test_replace_chars_given_probs_with_no_replacement(choices_mock):
    text_problems_emulator = TextProblemsEmulator()
    text_problems_emulator.text = "ggqq99"
    replacements = {
        "g": [["q", "9"], [.5, .3]],
    }
    text_problems_emulator.replace_chars_given_probs(replacements)
    assert text_problems_emulator.text == "ggqq99"
'''
