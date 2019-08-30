.. _cluster_instructions:

Instructions and examples for running jobs on Kale or Ukko2
-----------------------------------------------------------

Official instructions
^^^^^^^^^^^^^^^^^^^^^

`Kale <https://wiki.helsinki.fi/display/it4sci/Kale+User+Guide>`_ 

`Ukko2 <https://wiki.helsinki.fi/display/it4sci/Ukko2+User+Guide>`_

Example jobs
^^^^^^^^^^^^

The following commands need to be run every time you log in to one of the clusters:

.. code-block:: bash

    module load Python/3.7.0-intel-2018b
    export SCIKIT_LEARN_DATA=$TMPDIR

    cd $WRKDIR/dpEmu
    source venv/bin/activate

Running text classification example
"""""""""""""""""""""""""""""""""""

Create the batch file for the job:

.. code-block:: bash

    nano text_classification.job

Then write the following content to it and save the file. **Remember to put your username in place of <username>**:

.. code-block:: bash

    #!/bin/bash
    #SBATCH -J dpEmu
    #SBATCH --workdir=/wrk/users/<username>/dpEmu/
    #SBATCH -o text_classification_results.txt
    #SBATCH -c 8
    #SBATCH --mem=64G
    #SBATCH -t 10:00

    srun python3 examples/run_text_classification_example.py all 10
    srun sleep 60

Submit the batch job to be run:

.. code-block:: bash

    sbatch text_classification.job

You can view the execution of the code as if it was executed on your home terminal with:

.. code-block:: bash

    tail -f text_classification_results.txt

The resulting images will saved to the dpEmu/out directory.

Running object detection example
""""""""""""""""""""""""""""""""

First remember to load the required modules and install the object detection example requirements while in the virtual enviroment, if not done already:
:ref:`object_detection_cluster_requirements`.

Create the batch file for the job:

.. code-block:: bash

    nano object_detection.job

Then write the following content to it and save the file. **Remember to put your username in place of <username>**:

.. code-block:: bash

    #!/bin/bash
    #SBATCH -J dpEmu
    #SBATCH --workdir=/wrk/users/<username>/dpEmu/
    #SBATCH -o object_detection_results.txt
    #SBATCH -c 4
    #SBATCH --mem=32G
    #SBATCH -p gpu
    #SBATCH --gres=gpu:1
    #SBATCH -t 10:00:00

    srun python3 examples/run_object_detection_example.py
    srun sleep 60

Running this example can take a lot of time. You could try to disable some of the slowest models i.e. FasterRCNN and RetinaNet. To further speed up the job on Kale, by using the latest GPUs, add the following line to the batch file:

.. code-block:: bash

    #SBATCH --constraint=v100

Submit the batch job to be run:

.. code-block:: bash

    sbatch object_detection.job

You can view the execution of the code as if it was executed on your home terminal with:

.. code-block:: bash

    tail -f object_detection_results.txt

The resulting images will saved to the dpEmu/out directory.

Running object detection notebook
"""""""""""""""""""""""""""""""""

In the batch file replace:

.. code-block:: bash

    srun python3 examples/run_object_detection_example.py

with for example:

.. code-block:: bash

    srun jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --inplace --execute docs/case_studies/Object_Detection_JPEG_Compression.ipynb
