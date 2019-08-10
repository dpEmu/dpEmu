Installing dpEmu
================

To install dpEmu on your computer, run the following commands in your terminal:

.. code-block:: bash

    git clone https://github.com/dpEmu/dpEmu.git
    cd dpEmu
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements/base.txt (pip install -r requirements/with_examples.txt)
    pip install -e "git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
    pip install -e .
