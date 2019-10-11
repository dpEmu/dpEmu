# dpEmu

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![Build Status](https://travis-ci.com/dpEmu/dpEmu.svg?branch=master)](https://travis-ci.com/dpEmu/dpEmu)
[![codecov](https://codecov.io/gh/dpEmu/dpEmu/branch/master/graph/badge.svg)](https://codecov.io/gh/dpEmu/dpEmu)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/87b3b421702b4885a37f4025b59f5381)](https://www.codacy.com/app/thalvari/dpEmu?utm_source=github.com&utm_medium=referral&utm_content=dpEmu/dpEmu&utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/dpemu/badge/?version=latest)](https://dpemu.readthedocs.io/en/latest/?badge=latest)

dpEmu is a Python library for emulating data problems in the use and training of machine learning systems. It provides tools for injecting errors into data, running machine learning models with different error parameters and visualizing the results.

dpEmu is being built on the specifications and requirements provided by Professors Jukka K. Nurminen and Tommi Mikkonen for the course Software Engineering Lab at the University of Helsinki, department of Computer Science.


## Table of Contents

* [Installation](#installation)
* [Documentation](#documentation)
* [Research](#research)
* [Contributing](#contributing)
* [Credits](#credits)
* [Coursework Documentation](#coursework-documentation)

## Installation

**Note:** Works only with Python versions 3.6 and 3.7.

* `git clone https://github.com/dpEmu/dpEmu.git`
* `cd dpEmu`
* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip install -U pip setuptools wheel`
* `pip install -r requirements/base.txt`
* `pip install -e "git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
* `pip install -e .`

#### Additional step to run all of the examples

* `pip install -r requirements/examples.txt`

#### Additional steps for the object detection example and notebooks (CUDA and cuDNN required)

* `git clone https://github.com/dpEmu/Detectron.git libs/Detectron`
* `./scripts/install_detectron.sh`
* `git clone https://github.com/dpEmu/darknet.git libs/darknet`
* `./scripts/install_darknet.sh`

## Documentation

* [dpEmu Documentation](https://dpemu.readthedocs.io/en/latest/index.html)

## Research

Further information of our work is available in workshop paper. Please cite it if you are using our research.

*Jukka K. Nurminen, Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen, and Tommi Mikkonen. ["Software Framework for Data Fault Injection to Test Machine Learning Systems"](https://dl.dropboxusercontent.com/s/oz81eckesoy0t74/Data_Error_Injection_for_RSDA.pdf). 4th IEEE International Workshop on Reliability and Security Data Analysis (RSDA 2019) at 30th Annual IEEE International Symposium on Software Reliability Engineering (ISSRE 2019), Berlin, Germany, November 2019 (to appear)*.

## Contributing

See [Wiki](https://github.com/dpEmu/dpEmu/wiki/Contributing).

## Credits

* [Antti Röyskö](https://github.com/anroysko)
* [Juuso Silvennoinen](https://github.com/Jsos17)
* [Juha Mylläri](https://github.com/juhamyllari)
* [Juha Harviainen](https://github.com/Kalakuh)
* [Tuomas Halvari](https://github.com/thalvari)
  
## Coursework Documentation

* [Product Backlog](https://docs.google.com/spreadsheets/d/1WarfjE1UKnpkwlG3px8kG7dWvZmzVhzRg8-vwbMKG6c)
* [Definition of Done](coursework_docs/definition_of_done.md)
* [Meeting Etiquette](coursework_docs/meeting_etiquette.md)
* [Team Practices](coursework_docs/team_practices.md)
