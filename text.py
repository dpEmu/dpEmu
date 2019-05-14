import os
import random
from copy import deepcopy


class TextProblemsEmulator:
    def __init__(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def __replace(c, replacements):
        if c not in replacements:
            return c
        chars, probs = deepcopy(replacements[c])
        chars.append(c)
        probs.append(1 - sum(probs))
        return random.choices(chars, probs)[0]

    def replace_chars_given_probs(self, filename, replacements):
        text_path = os.path.join(self.folder, "data/" + filename)
        text = open(text_path, "r").read()
        result = "".join([self.__replace(c, replacements) for c in text])
        result_path = os.path.join(self.folder, "out/" + filename)
        result_file = open(result_path, "w+")
        result_file.write(result)
        result_file.close()


if __name__ == "__main__":
    text_problems_emulator = TextProblemsEmulator()
    replacements_1 = {
        "g": [["q", "9"], [.5, .3]],
    }
    text_problems_emulator.replace_chars_given_probs("word_sample_text.txt", replacements_1)
