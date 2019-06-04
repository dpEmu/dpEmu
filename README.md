# dpEmu

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![Build Status](https://travis-ci.com/dpEmu/dpEmu.svg?branch=master)](https://travis-ci.com/dpEmu/dpEmu)
[![codecov](https://codecov.io/gh/dpEmu/dpEmu/branch/master/graph/badge.svg)](https://codecov.io/gh/dpEmu/dpEmu)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/87b3b421702b4885a37f4025b59f5381)](https://www.codacy.com/app/thalvari/dpEmu?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dpEmu/dpEmu&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

dpEmu is being built on the specifications and requirements provided by Professors Jukka Nurminen and Tommi Mikkonen for the course Software Engineering Lab at the University of Helsinki, department of Computer Science.

The aim of the project is to build a Python library and an UNIX tool for emulating data problems in use, and training of machine learning systems.

## Table of Contents <a name="table-of-contents"/>
*  [Installation](#installation)
*  [Usage](#usage)
*  [Contributing](#contributing)
*  [Credits](#credits)
*  [Documentation](#documentation)

## Installation <a name="installation"/>
### Linux
*  ```git clone git@github.com:dpEmu/dpEmu.git```
*  ```cd dpEmu```
*  ```python3 -m venv venv```
*  ```source venv/bin/activate```
*  ```pip install -U pip setuptools wheel```
*  ```pip install -r requirements.txt```

### Windows
*  ```git clone git@github.com:dpEmu/dpEmu.git```
*  ```cd dpEmu```
*  ```python -m venv venv```
*  ```source venv/Scripts/activate```
*  ```python -m pip install -U pip setuptools wheel```
*  ```pip install -r requirements.txt```

## Usage <a name="usage"/>
* Run the program with ```python run.py config/example_commands.txt config/example_text_error_params.json config/example_combiner_config.json config/example_model_config.json```
* Run examples with commands like ```python3 -m src.examples.run_ocr_error_example```

## Contributing <a name="contributing"/>
See [Wiki](https://github.com/dpEmu/dpEmu/wiki/Contributing).

## Credits <a name="credits"/>
*  [Antti Röyskö](https://github.com/anroysko)
*  [Juuso Silvennoinen](https://github.com/Jsos17)
*  [Juha Mylläri](https://github.com/juhamyllari)
*  [Juha Harviainen](https://github.com/Kalakuh)
*  [Elizabeth Berg](https://github.com/reykjaviks)
*  [Tuomas Halvari](https://github.com/thalvari)

## Documentation <a name="documentation"/>
*  [Product Backlog](https://docs.google.com/spreadsheets/d/1WarfjE1UKnpkwlG3px8kG7dWvZmzVhzRg8-vwbMKG6c)
*  [Definition of Done](docs/definition_of_done.md)
*  [Meeting Etiquette](docs/meeting_etiquette.md)
*  [Team Practices](docs/team_practices.md)
*  [Sprint 0 Task Log](https://github.com/dpEmu/dpEmu/projects/1)
