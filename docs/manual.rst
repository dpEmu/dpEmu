User Manual
===========

Description
-----------

dpEmu is a Python library for emulating data problems in use and training of machine learning systems.

Installation
------------

To install dpEmu on your computer, run the following commands in your terminal:

.. code-block:: bash

    git clone git@github.com:dpEmu/dpEmu.git
    cd dpEmu
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    pip install pycocotools

You need to run also the following commands if you want to run the object detection example:

.. code-block:: bash

    git clone git@github.com:dpEmu/Detectron.git libs/Detectron
    ./scripts/install_detectron.sh
    git clone git@github.com:dpEmu/darknet.git libs/darknet
    ./scripts/install_darknet.sh

Usage
-----

dpEmu consists of three components:

* A system for building an error generator
* A system for running the AI models with different error parameters
* Tools for visualizing the results

Error generation
^^^^^^^^^^^^^^^^

First an error generation tree needs to be created. This is done by using the ``src.problemgenerator`` module which contains different kinds of tree nodes and filters for adding error to the data.

There are three generic node types, ``Array``, ``Series`` and ``TupleSeries``, and special node types for manipulating specific data types.

``Array`` node is used for handling any n-dimensional data and filters are directly applied to this data.

``Series`` node is given a child node as a parameter and when error is being generated it removes the outermost dimension of the data and passes each element to the child node
i.e. if a ``Series`` ``s`` has a child ``Array`` ``a`` and ``s`` is given an array ``[[1, 2], [3, 4]]`` as its input, then it passes arrays ``[1, 2]`` and ``[3, 4]`` to ``a``.

``TupleSeries`` node works quite similarly to ``Series``: it is given an array of child nodes and it removes the outermost dimension of the data 
and passes the first element to the first child, the second element to the second child and so on.

Filters can be added to ``Array`` nodes and they are used for manipulating data which can be images, time series, sound or something completely different. The ``filters.py`` file contains dozens of filters (e.g. ``Snow``, ``Blur`` and ``SensorDrift``) 
for these purposes and they can be added to an array node by using the ``addfilter`` function.

The parameters for the filters are given via a ``dict`` object when the error is being generated. During the initialization the filters are given the keys which are 
later used for getting the parameters.

Here is an example of what the error generation process might look like:

.. code-block:: python
    :linenos:

    # Assume our data is a tuple of the form (x, y) where x has
    # shape (100, 10) and y has shape (100,). We can think of each
    # row i as a data point where x_i represents the values of the
    # explanatory variables and y_i represents the corresponding
    # value of the response variable.
    x = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    data = (x, y)

    # Build a data model tree.
    x_node = array.Array()
    y_node = array.Array()
    root_node = series.TupleSeries([x_node, y_node])

    # Suppose we want to introduce NaN values (i.e. missing data)
    # to y only (thus keeping x intact).
    probability = .3
    y_node.addfilter(filters.Missing("p"))

    # Feed the data to the root node.
    output = root_node.generate_error(data, {"p": probability})

In the example the error generation tree has a ``TupleSeries`` as its root node, and it has two ``Array`` nodes as its children. Then on the line 18 we add a ``Missing`` filter to one of the children, 
which will transform some of the values in the 2-dimensional array ``y`` to NaN. The filter is given a parameter with value *"p"*, which means that the key for the probability for transforming a number into NaN is going to be *"p"* in the parameter dictionary.

Finally we call the ``generate_error`` function of the root node with the parameter *'p'* being 0.3, after which the function then returns the errorified data. However this part is usually done by and AI runner system, 
which we are going to discuss next.

AI runner system
^^^^^^^^^^^^^^^^

The AI runner system, or simply runner, is a system which is used for running multiple AI models simultaneously with distinct filter error parameters by using multithreading. After running all the models with all wanted parameter combinations 
the system returns a ``pandas.DataFrame`` object which can later be used for visualizing the results.

The runner needs to be given the following values when it is run: train data, test data, a preprocessor, an error generation tree, a list of error parameters, a list of AI models and their parameters and a boolean about whether to use interactive mode or not.

Train data and test data
""""""""""""""""""""""""
These are the original train data and test data which will be given to the AI models. A value ``None`` can also be passed to the runner if there is no train data.

Preprocessor
""""""""""""

The preprocessor needs to implement a function ``run(train_data, test_data)`` and it returns the preprocessed train and test data. The preprocessor can return additional data as well, and it will be listed as separate columns in the ``DataFrame`` which the runner returns.
Here is a simple example of a preprocessor, which does nothing to the original data, but returns also an array called *"negative_data"* which contains the additive inverse of each test_data's element.

.. code-block:: python
    :linenos:
    
    class Preprocessor:
        def __init__(self):
            self.random_state = RandomState(42)

        def run(self, train_data, test_data):
            negative_data = -test_data
            return train_data, test_data, {"negative_data": negative_data}

Error generation tree
"""""""""""""""""""""

The root node of the error generation tree should be given to the runner. The structure of the error generation tree is described above.

Error parameter list
""""""""""""""""""""

The list of error parameters is simply a list of dictionaries which contain the keys and error values for the error generation tree.

AI model parameter list
"""""""""""""""""""""""

The list of AI model parameters is a list of dictionaries containing three keys: *"model"*, *"params_list"* and *"use_clean_train_data"*. 

The value of *"model"* is **a class instead of an object**. 
The given class should implement the function ``run(train_data, test_data, parameters)`` which runs the model on the train data and test data with given parameters and returns a dictionary containing the scores and possibly additional data.

The value of *"params_list"* is a list of dictionaries where each dictionary contains one set of parameters for model. The model will be given these parameters when the ``run(train_data, test_data, parameters)`` function is called.

If the *"use_clean_train_data"* boolean is true, then no error will be added to the train data.

Here is an example AI model parameter list and a model:

.. code-block:: python
    :linenos:

    from numpy.random import RandomState 
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score

    # Model
    class KMeansModel:
        def __init__(self):
            self.random_state = RandomState(42)

        def run(self, train_data, test_data, model_params):
            labels = model_params["labels"]

            n_classes = len(np.unique(labels))
            fitted_model = KMeans(n_clusters=n_classes,
                                  random_state=self.random_state
                           ).fit(test_data)

            return {
                "AMI": round(adjusted_mutual_info_score(labels, 
                                                        fitted_model.labels_,
                                                        average_method="arithmetic"),
                             3),
                "ARI": round(adjusted_rand_score(labels, fitted_model.labels_), 3),
            }

    # Parameter list
    model_params_dict_list = [
        {"model": KMeansModel, "params_list": [{"labels": labels}]}
    ]

Interactive mode
""""""""""""""""

The final parameter of the runner system is a boolean telling whether to use interactive mode or not.
Some of the functions for visualizing the results require the interactive mode, for some of them it's optional
and most of them have no interactive functionality.

Basically what the interactive mode does is that it adds a column containing the modified test data to the resulting ``DataFrame`` object.
The interactive visualizer functions use this data to display points of data so that e.g. the programmer can try to figure out why
something was classified incorrectly.

Visualization functions
^^^^^^^^^^^^^^^^^^^^^^^

The module ``src.plotting`` has a file ``utils.py`` which contains multiple functions for plotting and visualizing the data.

Example
-------

Here is an unrealistic but simple example which demonstrates all three components of dpEmu. In this example we are trying to predict 
the next value of data when we know all earlier values in the data. Our model tries to do estimate this by keeping a weighted average.
In the end of the example a plot of scores is visualized.

.. code-block:: python
    :linenos:

    import sys

    import matplotlib.pyplot as plt
    import numpy as np

    from src import runner_
    from src.plotting.utils import visualize_scores
    from src.problemgenerator.array import Array
    from src.problemgenerator.filters import GaussianNoise


    class Preprocessor:
        def run(self, train_data, test_data):
            # Return the original data without preprocessing
            return train_data, test_data, {}


    class PredictorModel:
        def run(self, train_data, test_data, params):
            # The model tries to predict the values of test_data
            # by using a weighted average of previous values
            estimate = 0
            squared_error = 0

            for elem in test_data:
                # Calculate error
                squared_error += (elem - estimate) * (elem - estimate)
                # Update estimate
                estimate = (1 - params["weight"]) * estimate + params["weight"] * elem

            mean_squared_error = squared_error / len(test_data)

            return {"MSE": mean_squared_error}


    def main(argv):
        # Create some fake data
        if len(argv) == 2:
            train_data = None
            test_data = np.arange(int(sys.argv[1]))
        else:
            exit(0)

        # Create error generation tree that has an Array node
        # as its root node and a GaussianNoise filter
        err_root_node = Array()
        err_root_node.addfilter(GaussianNoise("mean", "std"))

        # The standard deviation goes from 0 to 20
        err_params_list = [{"mean": 0, "std": std} for std in range(0, 21)]

        # The model is run with different weighted estimates
        model_params_dict_list = [{
            "model": PredictorModel,
            "params_list": [{'weight': w} for w in [0.0, 0.05, 0.15, 0.5, 1.0]],
            "use_clean_train_data": False
        }]

        # Run the whole thing and get DataFrame for visualization
        df = runner_.run(train_data,
                        test_data,
                        Preprocessor,
                        err_root_node,
                        err_params_list,
                        model_params_dict_list,
                        use_interactive_mode=True)

        # Visualize mean squared error for all used standard deviations
        visualize_scores(df, ["MSE"], [False], "std", "Mean squared error")
        plt.show()


    if __name__ == "__main__":
        main(sys.argv)

Here's what the resulting image should look like:

.. image:: manual_demo.png
