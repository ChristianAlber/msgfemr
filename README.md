# msgfemr
msgfemr is a repository containing the implementation of methods described in our paper [reference to be added]. 
This repository includes scripts and tools for reproducing the figures and results presented in the publication.

## Features

- Modular codebase for numerical experiments.
- Example scripts (`ex_...`) to reproduce figures and results.
- Environment setup via conda for reproducibility.

## Installation

To set up the project environment, ensure you have [conda](https://docs.conda.io/en/latest/) installed. 
Then, create and activate the environment using the provided `requirements.yml` file:

```bash
conda env create -f requirements.yml
conda activate msgfemr
```

This will install all required dependencies.

## Usage

The repository contains example scripts prefixed with `ex_...` that demonstrate how to reproduce the figures from the paper. To run an example script, use:

```bash
python ex_channel.py
python ex_skyscraper.py
```

Each script is documented with comments explaining its purpose and usage. For more information about the experiments and expected outputs, refer to the paper.
The figures and tables created from the scripts will be saved in the folders plots and tables. 


## Citation

If you use this repository in your research, please cite our paper (details to be added).