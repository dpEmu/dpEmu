dpEmu
=====

|Open Source Love| |Build Status| |codecov| |Codacy Badge| |License:
MIT| |Documentation Status|

dpEmu is being built on the specifications and requirements provided by
Professors Jukka Nurminen and Tommi Mikkonen for the course Software
Engineering Lab at the University of Helsinki, department of Computer
Science.

The aim of the project is to build a Python library and an UNIX tool for
emulating data problems in use, and training of machine learning
systems.

.. _table-of-contents-:

Table of Contents 
------------------

-  `Installation`_
-  `Usage`_
-  `Contributing`_
-  `Credits`_
-  `Documentation`_

.. _installation-:

Installation 
-------------

-  ``git clone git@github.com:dpEmu/dpEmu.git``
-  ``cd dpEmu``
-  ``python3 -m venv venv``
-  ``source venv/bin/activate``
-  ``pip install -U pip setuptools wheel``
-  ``pip install -r requirements.txt``
-  ``pip install pycocotools``
-  ``pip install -e git+https://github.com/dpEmu/Detectron.git#egg=detectron``
-  ``ln -s $PWD/data venv/src/detectron/detectron/datasets/data/coco``

.. _usage-:

Usage 
------

-  Run example format: ``python -m src.examples.run_ocr_error_example``

.. _contributing-:

Contributing 
-------------

See `Wiki`_.

.. _credits-:

Credits 
--------

-  `Antti Röyskö`_
-  `Juuso Silvennoinen`_
-  `Juha Mylläri`_
-  `Juha Harviainen`_
-  `Elizabeth Berg`_
-  `Tuomas Halvari`_

.. _documentation-:

Documentation 
--------------

-  `Product Backlog`_
-  `Definition of Done`_
-  `Meeting Etiquette`_
-  `Team Practices`_

.. _Installation: #installation
.. _Usage: #usage
.. _Contributing: #contributing
.. _Credits: #credits
.. _Documentation: #documentation
.. _Wiki: https://github.com/dpEmu/dpEmu/wiki/Contributing
.. _Antti Röyskö: https://github.com/anroysko
.. _Juuso Silvennoinen: https://github.com/Jsos17
.. _Juha Mylläri: https://github.com/juhamyllari
.. _Juha Harviainen: https://github.com/Kalakuh
.. _Elizabeth Berg: https://github.com/reykjaviks
.. _Tuomas Halvari: https://github.com/thalvari
.. _Product Backlog: https://docs.google.com/spreadsheets/d/1WarfjE1UKnpkwlG3px8kG7dWvZmzVhzRg8-vwbMKG6c
.. _Definition of Done: docs/definition_of_done.md
.. _Meeting Etiquette: docs/meeting_etiquette.md
.. _Team Practices: docs/team_practices.md

.. |Open Source Love| image:: https://badges.frapsoft.com/os/v1/open-source.svg?v=103
   :target: https://github.com/ellerbrock/open-source-badges/
.. |Build Status| image:: https://travis-ci.com/dpEmu/dpEmu.svg?branch=master
   :target: https://travis-ci.com/dpEmu/dpEmu
.. |codecov| image:: https://codecov.io/gh/dpEmu/dpEmu/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/dpEmu/dpEmu
.. |Codacy Badge| image:: https://api.codacy.com/project/badge/Grade/87b3b421702b4885a37f4025b59f5381
   :target: https://www.codacy.com/app/thalvari/dpEmu?utm_source=github.com&utm_medium=referral&utm_content=dpEmu/dpEmu&utm_campaign=Badge_Grade
.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
.. |Documentation Status| image:: https://readthedocs.org/projects/dpemu/badge/?version=latest
   :target: https://dpemu.readthedocs.io/en/latest/?badge=latest
