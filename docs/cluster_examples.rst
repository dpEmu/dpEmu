.. _cluster_examples:

Example jobs on Kale and Ukko2
------------------------------

Running  text classification example on Kale or Ukko2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    module load Python/3.7.0-intel-2018b
    export SCIKIT_LEARN_DATA=$TMPDIR

    cd $WRKDIR/dpEmu
    source venv/bin/activate


Create the batch file for the job:

.. code-block:: bash

    nano batch-submit.job

Then write the following content to it and save the file. **Remember to put your username in place of <username>**:

.. code-block:: bash

    #!/bin/bash
    #SBATCH -J dpEmu
    #SBATCH --workdir=/wrk/users/<username>/dpEmu/
    #SBATCH -o text_classification_result.txt
    #SBATCH -c 8
    #SBATCH --mem=128G
    #SBATCH -t 20:00

    srun python3 -m src.examples.run_text_classification_example all 20
    srun sleep 60

Submit the batch job to be run:

.. code-block:: bash

    sbatch batch-submit.job

You can view the execution of the code as if it was executed on your home terminal:

.. code-block:: bash

    tail -f text_classification_result.txt

The example src.examples.run_text_classification_example will save images to the dpEmu/out directory.

Running object detection example on Kale or Ukko2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remember to clone the relevant repositorios and run the required scripts, if you have not already. See `Object detection example requirements`_

.. code-block:: bash

    module load CUDA/10.0.130
    module load cuDNN/7.5.0.56-CUDA-10.0.130
    module load Python/3.7.0-intel-2018b

    cd $WRKDIR/dpEmu
    source venv/bin/activate

Create the batch file for the job:

.. code-block:: bash

    nano batch-submit.job

Then write the following content to it and save the file. **Remember to put your username in place of <username>**:

.. code-block:: bash

    #!/bin/bash
    #SBATCH -J dpEmu
    #SBATCH --workdir=/wrk/users/<username>/dpEmu/
    #SBATCH -o object_detection_example.txt
    #SBATCH -c 4
    #SBATCH --mem=32G
    #SBATCH -p gpu
    #SBATCH --gres=gpu:1
    #SBATCH -t 10:00:00

    srun python3 -m src.examples.run_object_detection_example
    srun sleep 60

Submit the batch job to be run:

.. code-block:: bash

    sbatch batch-submit.job

You can view the execution of the code as if it was executed on your home terminal:

.. code-block:: bash

    tail -f object_detection_example.txt
