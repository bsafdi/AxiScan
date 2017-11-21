# AxiScan

**A Likelihood Framework for Axion Direct Detection**

[![arXiv](https://img.shields.io/badge/arXiv-1711.0xxxx%20-green.svg)](https://arxiv.org/abs/1711.0xxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Sensitivity](https://github.com/bsafdi/AxiScan/blob/master/examples/Projected_Sensitivity.png "Projected sensitivity versus S/N=1")

AxiScan is a repository of tools for implementing a likelihood framework for axion direct detection experiments. It is designed to replace traditional signal to noise based estimates, as shown in the figure above. If this code helps contribute to published work, please cite the original reference [1711.0xxxx](https://arxiv.org/abs/1711.0xxxx). 

## Authors

- Joshua Foster; fosterjw at umich dot edu
- Nicholas Rodd; nrodd at mit dot edu
- Benjamin Safdi; bsafdi at umich dot edu

## Installation

AxiScan is written in a combination of `python` and `cython`. To install the codebase along with all its dependencies, use the setup script

```
$ python setup.py install
```

The setup script also compile the cython. To manually compile the cython locally, execute the following 

```
$ make build
```

## Examples

Worked examples can be found [here](https://github.com/bsafdi/AxiScan/tree/master/examples). Before running these, the cython should be installed and then compiled as follows: within the `AxiScan/` folder, execute `make.sh`.
