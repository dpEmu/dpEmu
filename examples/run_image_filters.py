# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dpemu.nodes import Array
from dpemu.filters.image import Blur, BlurGaussian, Brightness, JPEG_Compression, \
    LensFlare, Rain, Resolution, Rotation, Saturation, Snow, StainArea

fig = plt.figure(figsize=(8, 8))


def visualize(title, ind, data, flt, params):
    root_node = Array()
    root_node.addfilter(flt)
    result = root_node.generate_error(data, params)
    sub = fig.add_subplot(3, 4, ind + 1)
    sub.imshow(result)
    sub.set_title(title)


def applyBlur(data, ind):
    visualize("Blur", ind, data, Blur("rep", "rad"), {"rep": 3, "rad": 10})


def applyBlurGaussian(data, ind):
    visualize("Gaussian Blur", ind, data, BlurGaussian("std"), {"std": 5})


def applyBrightness(data, ind):
    visualize("Brightness", ind, data, Brightness("tar", "rat", "range"), {"tar": 1, "rat": 0.5, "range": 255})


def applyJPEGCompression(data, ind):
    visualize("JPEG Compression", ind, data, JPEG_Compression("qual"), {"qual": 3})


def applyLensFlare(data, ind):
    visualize("Lens Flare", ind, data, LensFlare(), {})


def applyRain(data, ind):
    visualize("Rain", ind, data, Rain("prob", "range"), {"prob": 0.005, "range": 255})


def applyResolution(data, ind):
    visualize("Resolution", ind, data, Resolution("k"), {"k": 10})


def applyRotation(data, ind):
    visualize("Rotation", ind, data, Rotation("min_a", "max_a"), {"min_a": -30, "max_a": 30})


def applySaturation(data, ind):
    visualize("Saturation", ind, data, Saturation("tar", "rat", "range"), {"tar": 0, "rat": 0.5, "range": 255})


def applySnow(data, ind):
    visualize("Snow", ind, data, Snow("prob", "flake_a", "backg_a"), {"prob": 0.001, "flake_a": 0.9, "backg_a": 1})


def applyStainArea(data, ind):
    class rd_gen():
        def generate(self, random_state):
            return random_state.randint(50, 100)
    visualize("Stain Area", ind, data, StainArea("prob", "rd_gen", "tsp"),
              {"prob": 0.00005, "rd_gen": rd_gen(), "tsp": 0})


def main():
    img = Image.open("data/landscape.png")
    data = np.array(img)

    applyBlur(data, 0)
    applyBlurGaussian(data, 1)
    applyBrightness(data, 2)
    applyJPEGCompression(data, 3)
    applyLensFlare(data, 4)
    applyRain(data, 5)
    applyResolution(data, 6)
    applyRotation(data, 7)
    applySaturation(data, 8)
    applySnow(data, 9)
    applyStainArea(data, 10)

    plt.show()


if __name__ == "__main__":
    main()
