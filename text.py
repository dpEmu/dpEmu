import os
from random import random


class TextProblemsEmulator:
    def __init__(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def __rand_replace(c, probs):
        return probs[c][0] if c in probs and random() < probs[c][1] else c

    def replace_letters_with_probs(self, filename, probs):
        text_path = os.path.join(self.folder, "data/" + filename)
        text = open(text_path, "r").read()
        result = "".join([self.__rand_replace(c, probs) for c in text])
        result_path = os.path.join(self.folder, "out/" + filename)
        result_file = open(result_path, "w+")
        result_file.write(result)
        result_file.close()


if __name__ == "__main__":
    text_problems_emulator = TextProblemsEmulator()
    probs = {
        "f": ("t", .5)
    }
    text_problems_emulator.replace_letters_with_probs("word_sample_text.txt", probs)
