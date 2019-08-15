Installation on University of Helsinki clusters (Ukko2 and Kale)
----------------------------------------------------------------

First you need to have access rights to the clusters. See instructions for who can get access rights to `Kale <https://wiki.helsinki.fi/display/it4sci/Kale+User+Guide#KaleUserGuide-Access>`_ or to `Ukko2 <https://wiki.helsinki.fi/display/it4sci/Ukko2+User+Guide#Ukko2UserGuide-1.0Access>`_.

To install dpEmu on Kale or Ukko2 clusters, first establish a ssh connection to the cluster:

.. code-block:: bash

    ssh ukko2.cs.helsinki.fi

Or:

.. code-block:: bash

    ssh kale.grid.helsinki.fi

To install dpEmu without the ability of running all of the examples, execute the following commands in remote terminal:

.. code-block:: bash

    module load Python/3.7.0-intel-2018b
    export SCIKIT_LEARN_DATA=$TMPDIR

    cd $WRKDIR
    git clone https://github.com/dpEmu/dpEmu.git
    cd dpEmu
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip setuptools wheel --cache-dir $TMPDIR
    pip install -r requirements/base.txt --cache-dir $TMPDIR
    pip install -e "git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI" --cache-dir $TMPDIR
    pip install -e . --cache-dir $TMPDIR

In order to run all of the examples, you'll also need to execute the following command:

.. code-block:: bash

    pip install -r requirements/examples.txt --cache-dir $TMPDIR

.. _object_detection_requirements:

Object detection example requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Further installation steps and a NVIDIA GPU are needed to run the object detection example. Execute the following commands after all of the above:

.. code-block:: bash

    module load CUDA/10.0.130
    module load cuDNN/7.5.0.56-CUDA-10.0.130

    git clone https://github.com/dpEmu/Detectron.git libs/Detectron
    ./scripts/install_detectron.sh
    git clone https://github.com/dpEmu/darknet.git libs/darknet
    ./scripts/install_darknet.sh


:ref:`cluster_instructions`.
