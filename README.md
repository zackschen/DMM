# Not Like a Rolling Stone: Continual Referring Expression Comprehension via Dual Modular Memorization

## Abstract

Referring Expression Comprehension (REC) aims to localize an image region of a given object described by a natural-language expression. While promising performance has been demonstrated, existing REC algorithms make a strong assumption that training data feeding into a model are given upfront, which degrades its practicality for real-world scenarios. In this paper, we propose Continual Referring Expression Comprehension (CREC), a new setting for REC, where a model is learned on a stream of incoming tasks. In order to continuously improve the model on sequential tasks without forgetting prior learned knowledge and without repeatedly re-training from a scratch, we propose an effective baseline method named Dual Modular Memorization (DMM), which alleviates the problem of catastrophic forgetting by two memorization modules: Implicit-Memory and Explicit-Memory. Specifically, the former module aims to constrain drastic changes to important parameters learned on old tasks when learning a new task; while the latter module maintains a buffer pool to dynamically select and store representative samples of each seen task for future rehearsal. We create three benchmarks for the new CREC setting, by respectively re-splitting three widely-used REC datasets RefCOCO, RefCOCO+ and RefCOCOg into sequential tasks. Extensive experiments on the constructed benchmarks demonstrate that our DMM method significantly outperforms  other  alternatives, based on two  popular REC backbones.

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


