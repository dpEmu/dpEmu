.. _cluster_instructions:

Installation on University of Helsinki clusters (Ukko2 and Kale)
----------------------------------------------------------------

First you need to have access rights to the clusters. See instructions for who can get access rights to `Kale <https://wiki.helsinki.fi/display/it4sci/Kale+User+Guide#KaleUserGuide-Access>`_ or to `Ukko2 <https://wiki.helsinki.fi/display/it4sci/Ukko2+User+Guide#Ukko2UserGuide-1.0Access>`_.

To install dpEmu on Kale or Ukko2 clusters, first establish a ssh connection to the cluster:

.. code-block:: bash

    ssh ukko2.cs.helsinki.fi

Or:

.. code-block:: bash

    ssh kale.grid.helsinki.fi

Now you can install dpEmu by running the following commands in the remote terminal:

.. code-block:: bash

    module load Python/3.7.0-intel-2018b
    export SCIKIT_LEARN_DATA=$TMPDIR

    cd $WRKDIR
    git clone git@github.com:dpEmu/dpEmu.git
    cd dpEmu
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip setuptools wheel --cache-dir $TMPDIR
    pip install -r requirements.txt --cache-dir $TMPDIR
    pip install pycocotools --cache-dir $TMPDIR

Object detection example requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You also need to run the following commands if you want to run the object detection example:

.. code-block:: bash

    module load CUDA/10.0.130
    module load cuDNN/7.5.0.56-CUDA-10.0.130

    cd $WRKDIR/dpEmu
    source venv/bin/activate
    git clone git@github.com:dpEmu/Detectron.git libs/Detectron
    ./scripts/install_detectron.sh
    git clone git@github.com:dpEmu/darknet.git libs/darknet
    ./scripts/install_darknet.sh


Instructions for running jobs on Kale or Ukko2
----------------------------------------------

Official instructions: `Kale <https://wiki.helsinki.fi/display/it4sci/Kale+User+Guide>`_ or `Ukko2 <https://wiki.helsinki.fi/display/it4sci/Ukko2+User+Guide>`_

:ref:`cluster_examples`.
