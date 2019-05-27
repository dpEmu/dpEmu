import json
import io
import sys

from datetime import datetime

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

class Combiner:
    @staticmethod
    def __plot_to_img():
        """Converts a figure into a .png image"""

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_img = buf.read()
        byte_img = io.BytesIO(byte_img)
        return Image.open(byte_img)

    @staticmethod
    def __analyze_config(config):
        """Analyzes the configuration file and returns the important values"""

        paths = {}
        filter_type = config["filter_type"]
        plot_scores_to_same = True
        if "plot_scores_to_same_diagram" in config:
            plot_scores_to_same = config["plot_scores_to_same_diagram"]
        paths["filters"] = []
        paths["scores"] = []
        paths["images"] = []
        Combiner.__analyze_config_structure(config["structure"], paths, [])
        return paths, filter_type, plot_scores_to_same

    @staticmethod
    def __analyze_config_structure(config, paths, path):
        """Analyzes the structure element of the config file and calculates the paths to different values."""

        index = 0
        for elem in config:
            path.append(index)
            if isinstance(elem, list):
                Combiner.__analyze_config_structure(elem, paths, path)
            if isinstance(elem, str):
                if elem == "filters":
                    paths["filters"] = path.copy()
                elif elem == "scores":
                    paths["scores"] = path.copy()
                elif elem == "image":
                    paths["images"].append(path.copy())
                else:
                    print("Unknown value '", elem, "' in the configuration file of the combiner.")
            path.pop()
            index += 1

    @staticmethod
    def __get_value(element, path):
        """
        Returns the value from a n-dimensional structure from a given path.

        e.g. __get_value(((0, 1), (2, 3)), [1, 0]) == 2
        """

        if not path:
            return element
        return Combiner.__get_value(element[path[0]], path[1:])

    @staticmethod
    def __create_combined_image(data, output_path, config_paths):
        """
        Combines images given in each element of the data to a single image
        """

        if not output_path:
            return

        for elem in data:
            total_width = 0
            max_height = 0
            for image_path in config_paths["images"]:
                image = Combiner.__get_value(elem, image_path)
                max_height = max(max_height, image.size[1])
                total_width += image.size[0]

            # combine images to single images
            combined_image = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            draw = ImageDraw.Draw(combined_image)

            for image_path in config_paths["images"]:
                image = Combiner.__get_value(elem, image_path)
                combined_image.paste(image, (x_offset, 0))
                x_offset += image.size[0]

            filter_map = str(Combiner.__get_value(elem, config_paths["filters"]))
            draw.text((total_width / 2 - len(filter_map) * 3, max_height - 15), filter_map, fill=(0, 0, 0))

            if output_path:
                time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                Combiner.__save(combined_image, output_path + "/combined-" + time + ".png")

    @staticmethod
    def __save(image, path):
        print("Saving an image to " + path)
        image.save(path)
        Image.open(path).verify()

    @staticmethod
    def combine(data, output_path=None, config_path=None):
        """
        Plots the data and saves the graphs to the output_path if it's specified.

        A configuration file is required to explain the function the structure of the data.
        The .json file is expected to have the following structure:
        {
          "filter_type": "...",
          "structure": [...]
        }

        where structure defines the n-dimensional structure of the data.
        Each array can contain either arrays or strings. The following strings are allowed:

        "scores": this element of data contains a dictionary containing each score and its type
        "filters": this element of data contains a dictionary which contains the configuration values
                    of the filters and the names of the filters
        "image": this element of data contains an image
        """

        if not config_path:
            print("Configuration file for the combiner is not specified.")
            sys.exit()

        config = json.loads(open(config_path, 'r').read())
        config_paths, filter_type, plot_scores_to_same = Combiner.__analyze_config(config)

        score_types = []
        for element in data:
            for score in Combiner.__get_value(element, config_paths["scores"]):
                score_types.append(score)

        # force list to contain only unique elements
        score_types = list(set(score_types))

        # plot scores
        if plot_scores_to_same:
            fig = plt.figure()
            plt.clf()
            ax = fig.add_subplot(111)
            for score_type in score_types:
                filter_values = []
                scores = []
                for element in data:
                    filter_values.append(Combiner.__get_value(element, config_paths["filters"])[filter_type])
                    scores.append(float(Combiner.__get_value(element, config_paths["scores"])[score_type]))
                line, = ax.plot(filter_values, scores)
                line.set_label(score_type)
                ax.scatter(filter_values, scores)
                plt.legend()
            plt.title("Scores")
            plt.xlabel("error")
            plt.ylabel("score")
            plt.tight_layout()
            img = Combiner.__plot_to_img()
            img.show()
            if output_path:
                Combiner.__save(img, output_path + "/scores-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
        else:
            for score_type in score_types:
                filter_values = []
                scores = []
                for element in data:
                    filter_values.append(Combiner.__get_value(element, config_paths["filters"])[filter_type])
                    scores.append(Combiner.__get_value(element, config_paths["scores"])[score_type])
                plt.figure()
                plt.clf()
                plt.plot(filter_values, scores)
                plt.scatter(filter_values, scores)
                plt.title(score_type)
                plt.xlabel("error")
                plt.ylabel("score")
                plt.tight_layout()
                img = Combiner.__plot_to_img()
                img.show()
                if output_path:
                    path = output_path + "/" + score_type + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png"
                    Combiner.__save(img, path)

        Combiner.__create_combined_image(data, output_path, config_paths)
