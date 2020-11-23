# Dynamic Data Fusion Model

This is a matlab implementation of the data fusion framework as described in our paper:
"Dynamic Fusion of Eye Movement Data and Verbal Narrations in Knowledge-rich Domains" (NIPS 2020)

## Installation

1. Install the following dependencies.
* matlab 2019 (or newer)
* python 3.7 (or newer)
* scikit-learn
* scipy
* numpy
* pickle
2. Copy all files into your local environment.
3. Change the current directory for matlab and python.

## Run the demo

For the demo of split-merge-switch sampler on synthetic data, start matlab and run demo.m

For the demo of the proposed framework on eye movement and verbal narration data, start matlab and run main.m 

For the evaluation on supervised tasks, use the command: python eval.py

## Dataset

The data is stored in the following files: 
* NarrationMatrixNoEye.csv: stores verbal narration data as word indices
* EyeMatrix.csv: stores eye movement data as compact representations
* Dictionary.csv: stores actual words with corresponding indices
* Idx2PosMatrix.csv: stores the matching between two modals
* EyePlot.csv: stores eye movement data for visualization

## Question

If you have any question, please feel free to contact me via email. 

## Cite

Please cite our paper if you use this code in your own work:

```
@article{zheng2020dynamic,
  title={Dynamic Fusion of Eye Movement Data and Verbal Narrations in Knowledge-Rich Domains},
  author={Zheng, Ervine and Yu, Qi and Li, Rui and Shi, Pengcheng and Haake, Anne},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
