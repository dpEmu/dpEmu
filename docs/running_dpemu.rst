Running dpEmu
=============

Running notebooks
-----------------

All jupyter notebooks provided can be opened in a browser with:

.. code-block:: bash

    jupyter notebook docs/case_studies/Text_Classification_OCR_Error.ipynb

and remotely executed in console with:

.. code-block:: bash

    jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --inplace --execute docs/case_studies/Text_Classification_OCR_Error.ipynb


Running scripts
---------------

User defined scripts are run similarly as the predefined examples.

**Run the examples from project root.**

If the examples do not require command line arguments, then
they can be run as follows:

.. code-block:: bash

    python3 examples/run_saturation_example_rgb_0_to_1.py

If the examples require command line arguments, add them after
the name of the file, each one separated by space (the argument
22 tells the angle of the counterclockwise rotation of the picture):

.. code-block:: bash

    python3 examples/run_rotate_example.py 22

The interactive mode is used in some examples and is activated by writing ``-i``:

.. code-block:: bash

    python3 examples/run_text_classification_example test 4 -i
