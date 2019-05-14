import os
import random
from copy import deepcopy


class TextProblemsEmulator:
    def __init__(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.text = None
        self.result = None

    @staticmethod
    def __replace_char(c, replacements):
        if c not in replacements:
            return c
        chars, probs = deepcopy(replacements[c])
        chars.append(c)
        probs.append(1 - sum(probs))
        return random.choices(chars, probs)[0]

    def replace_chars_given_probs(self, replacements):
        self.result = "".join([self.__replace_char(c, replacements) for c in self.text])

    def read_text(self, filename):
        text_path = os.path.join(self.folder, "data/" + filename)
        self.text = open(text_path, "r").read()

    def write_text(self, filename):
        result_path = os.path.join(self.folder, "out/" + filename)
        result_file = open(result_path, "w+")
        result_file.write(self.result)
        result_file.close()


if __name__ == "__main__":
    text_problems_emulator = TextProblemsEmulator()
    replacements_1 = {
        "g": [["q", "9"], [.5, .3]],
    }
    text_problems_emulator.read_text("word_sample_text.txt")
    text_problems_emulator.replace_chars_given_probs(replacements_1)
    text_problems_emulator.write_text("word_sample_text.txt")
