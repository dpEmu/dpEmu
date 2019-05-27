import json
import io
import sys

from datetime import datetime

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

class Combiner:
    @staticmethod
    def __plot_to_img():
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_img = buf.read()
        byte_img = io.BytesIO(byte_img)
        return Image.open(byte_img)

    @staticmethod
    def __analyze_config(config):
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
        if not path:
            return element
        return Combiner.__get_value(element[path[0]], path[1:])

    @staticmethod
    def __create_combined_image(data, config_paths):
        total_width = 0
        total_height = 0
        largest_image_width_in_column = [] # i-th element contains the width of the largest image in the i-th column
        for elem in data:
            max_height = 0
            index = 0
            for image_path in config_paths["images"]:
                image = Combiner.__get_value(elem, image_path)
                if index == len(largest_image_width_in_column):
                    largest_image_width_in_column.append(image.size[0])
                else:
                    largest_image_width_in_column[index] = max(largest_image_width_in_column[index], image.size[0])
                max_height = max(max_height, image.size[1])
                index += 1
            total_height += max_height

        total_width = sum(largest_image_width_in_column)

        # combine images to single images
        combined_image = Image.new('RGB', (total_width, total_height))
        x_offset = 0
        y_offset = 0
        draw = ImageDraw.Draw(combined_image)
        for elem in data:
            index = 0
            max_height = 0
            for image_path in config_paths["images"]:
                image = Combiner.__get_value(elem, image_path)
                combined_image.paste(image, (x_offset, y_offset))
                x_offset += largest_image_width_in_column[index]
                max_height = max(max_height, image.size[1])
                index += 1
            y_offset += max_height
            x_offset = 0

            filter_map = str(Combiner.__get_value(elem, config_paths["filters"]))
            draw.text((total_width / 2 - len(filter_map) * 3, y_offset - 15), filter_map, fill=(0, 0, 0))

        combined_image.show()

    # Plots the data and saves the graphs to the output_path if it's specified.
    #
    # A configuration file is required to explain the function the structure of the data.
    # The .json file is expected to have the following structure:
    # {
    #   "filter_type": "...",
    #   "structure": [...]
    # }
    def combine(self, data, output_path=None, config_path=None):
        if not config_path:
            print("Configuration file for the combiner is not specified.")
            sys.exit()

        config = json.loads(open(config_path, 'r').read())
        config_paths, filter_type, plot_scores_to_same = self.__analyze_config(config)

        score_types = []
        for element in data:
            for score in self.__get_value(element, config_paths["scores"]):
                score_types.append(score)

        # force list to contain only unique elements
        score_types = list(set(score_types))

        # plot scores
        if plot_scores_to_same:
            plt.figure()
            plt.clf()

            for score_type in score_types:
                filter_values = []
                scores = []
                for element in data:
                    filter_values.append(self.__get_value(element, config_paths["filters"])[filter_type])
                    scores.append(self.__get_value(element, config_paths["scores"])[score_type])
                line, = plt.plot(filter_values, scores)
                line.set_label(score_type)
                plt.scatter(filter_values, scores)
                plt.legend()
            plt.title("Scores")
            plt.xlabel("error")
            plt.ylabel("score")
            plt.tight_layout()
            img = self.__plot_to_img()
            img.show()
            if output_path:
                img.save(output_path + "/scores-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
        else:
            for score_type in score_types:
                filter_values = []
                scores = []
                for element in data:
                    filter_values.append(self.__get_value(element, config_paths["filters"])[filter_type])
                    scores.append(self.__get_value(element, config_paths["scores"])[score_type])
                plt.figure()
                plt.clf()
                plt.plot(filter_values, scores)
                plt.scatter(filter_values, scores)
                plt.title(score_type)
                plt.xlabel("error")
                plt.ylabel("score")
                plt.tight_layout()
                img = self.__plot_to_img()
                img.show()
                if output_path:
                    path = output_path + "/" + score_type + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png"
                    img.save(path)

        self.__create_combined_image(data, config_paths)
