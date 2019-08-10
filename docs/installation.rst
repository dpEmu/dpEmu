Installing dpEmu
================

To install dpEmu on your computer, run the following commands in your terminal:

.. code-block:: bash

    git clone https://github.com/dpEmu/dpEmu.git
    cd dpEmu
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    pip install pycocotools
    pip install -e .
