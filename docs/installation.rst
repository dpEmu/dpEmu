Installing dpEmu
================

To install dpEmu on your computer without the ability of running examples, execute the following commands in your terminal:

.. code-block:: bash

    git clone https://github.com/dpEmu/dpEmu.git
    cd dpEmu
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements/base.txt
    pip install -e "git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
    pip install -e .



**Or**, in order to run the examples, you need to execute the following commands:

.. code-block:: bash

    git clone https://github.com/dpEmu/dpEmu.git
    cd dpEmu
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements/base.txt
    pip install -r requirements/with_examples.txt
    pip install -e "git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
    pip install -e .

Additionally, run the following command in order to locally build the documentation with Sphinx:

.. code-block:: bash

    pip install -r requirements/docs.txt
