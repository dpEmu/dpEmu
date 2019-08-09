# dpEmu

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![Build Status](https://travis-ci.com/dpEmu/dpEmu.svg?branch=master)](https://travis-ci.com/dpEmu/dpEmu)
[![codecov](https://codecov.io/gh/dpEmu/dpEmu/branch/master/graph/badge.svg)](https://codecov.io/gh/dpEmu/dpEmu)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/87b3b421702b4885a37f4025b59f5381)](https://www.codacy.com/app/thalvari/dpEmu?utm_source=github.com&utm_medium=referral&utm_content=dpEmu/dpEmu&utm_campaign=Badge_Grade)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Documentation Status](https://readthedocs.org/projects/dpemu/badge/?version=latest)](https://dpemu.readthedocs.io/en/latest/?badge=latest)

dpEmu is being built on the specifications and requirements provided by Professors Jukka Nurminen and Tommi Mikkonen for the course Software Engineering Lab at the University of Helsinki, department of Computer Science.

The aim of the project is to build a Python library and an UNIX tool for emulating data problems in use, and training of machine learning systems.

## Table of Contents

* [Installation](#installation)
* [User manual](#user-manual)
* [Contributing](#contributing)
* [Credits](#credits)
* [Documentation](#documentation)
* [Coursework Documentation](#coursework-documentation)

## Installation

* `git clone git@github.com:dpEmu/dpEmu.git`
* `cd dpEmu`
* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip install -U pip setuptools wheel`
* `pip install -r requirements/base.txt` (or `pip install -r requirements/with_examples.txt`)
* `pip install -e .`

#### Additional steps for object detection example (CUDA and cuDNN required)

* `git clone git@github.com:dpEmu/Detectron.git libs/Detectron`
* `./scripts/install_detectron.sh`
* `git clone git@github.com:dpEmu/darknet.git libs/darknet`
* `./scripts/install_darknet.sh`

## User Manual

* [User manual](https://dpemu.readthedocs.io/en/latest/manual.html)

## Contributing

See [Wiki](https://github.com/dpEmu/dpEmu/wiki/Contributing).

## Credits

* [Antti Röyskö](https://github.com/anroysko)
* [Juuso Silvennoinen](https://github.com/Jsos17)
* [Juha Mylläri](https://github.com/juhamyllari)
* [Juha Harviainen](https://github.com/Kalakuh)
* [Elizabeth Berg](https://github.com/reykjaviks)
* [Tuomas Halvari](https://github.com/thalvari)
  
## Documentation

* [dpEmu Documentation](https://dpemu.readthedocs.io/en/latest/index.html)
  
## Coursework Documentation

* [Product Backlog](https://docs.google.com/spreadsheets/d/1WarfjE1UKnpkwlG3px8kG7dWvZmzVhzRg8-vwbMKG6c)
* [Definition of Done](docs/definition_of_done.md)
* [Meeting Etiquette](docs/meeting_etiquette.md)
* [Team Practices](docs/team_practices.md)
