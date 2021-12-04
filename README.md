# Not Like a Rolling Stone: Continual Referring Expression Comprehension via Dual Modular Memorization

## Abstract

Referring Expression Comprehension (REC) aims to localize the image region of a given object described by a natural-language expression. While promising performance has been demonstrated, existing REC algorithms whimsically  assume that the training data feeding into the model are given upfront, which degrades its practicality for real-world scenarios. In this paper, we propose Continual Referring Expression Comprehension (CREC), a new setting for REC, where the model is  learned on a stream of incoming tasks. In order to continuously improve the model on the sequential tasks without forgetting prior knowledge, an effective baseline method dubbed Dual Modular Memorization (DMM) is developed for CREC, which alleviates the problem of catastrophic forgetting by two memorization modules: Implicit-Memory and Explicit-Memory. Specifically, the former module aims to constraint drastic changes to important parameters learned on old tasks when learning a new task; while the latter one maintains a buffer pool to dynamically select and store representative samples of each seen task for future rehearsal. We create three benchmarks for the new CREC setting, by respectively re-splitting the three REC datasets RefCOCO, RefCOCO+ and RefCOCOg into sequential tasks. Extensive experiments on  the constructed benchmarks  demonstrate   that our DMM significantly  outperforms  other  alternatives  based on  two  popular  REC  backbones.

## Prerequisites
- Python 2.7
- Pytorch 0.2 (may not work with 1.0 or higher)
- CUDA 8.0

## Framework

![](https://raw.githubusercontent.com/zackschen/PictureBed/master/20211103211052.png)



## Setup
The code is implemented on https://github.com/lichengunc/MAttNet. Follow the instructions in it.

## Help

Feel free to ping me ([cczacks@gmail.com](mailto:cczacks@gmail.com)) if you encounter trouble getting it to work!

## Bibtex


