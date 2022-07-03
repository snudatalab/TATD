# TATD

This is a PyTorch and TensorLy implementation of **Time-Aware Tensor Decomposition for Sparse Tensors**.<br>


## Prerequisites

- Python 3.6+
- [tqdm](https://tqdm.github.io)
- [NumPy](https://numpy.org)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [TensorLy](http://tensorly.org/stable/index.html)


## Usage

- Install all of the prerequisites
- You can run the demo script by `bash demo.sh`, which simply runs `src/main.py`.
- You can change the datasets and hyper-parameters by modifying `src/main.py`.
- you can check out the running results in `out` directory.


## Datasets
Preprocessed datasets are in the `data` directory.

|         Name        |          Description          |      Size      |   NNZ  | Granularity in Time |                                    Original Source                                   |
|:-------------------:|:-----------------------------:|:--------------:|:------:|:-------------------:|:------------------------------------------------------------------------------------:|
| Beijing Air Quality | time x locations x pollutants | 35064 x 12 x 6 | 2454305 | hourly              | [Link](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Datal) |
| Madrid Air Quality  | time x locations x pollutants | 2678 x 26 x 17 | 337759  | daily               | [Link](https://www.kaggle.com/decide-soluciones/air-quality-madrid)                  |
| Radar Traffic       | time x locations x directions | 17937 x 17 x 5 | 495685  | hourly              | [Link](https://www.kaggle.com/vinayshanbhag/radar-traffic-data)                      |
| Indoor Condition    | time x locations x sensor     | 19735 x 9 x 2  | 241201  | every 10 minutes    | [Link](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)         |
| Server Room         | time x air conditioning x server power x locations  | 4157 x 3 x 3 x 34  | 1009426 | 1 second    | [Link](https://zenodo.org/record/3610078#.XlNpAigzaM8)                                 |

## Reference

If you use this code, please cite the following paper.

```
@article{DBLP:journals/ml/AhnJK22,
  author    = {Dawon Ahn and
               Jun{-}Gi Jang and
               U Kang},
  title     = {Time-aware tensor decomposition for sparse tensors},
  journal   = {Mach. Learn.},
  volume    = {111},
  number    = {4},
  pages     = {1409--1430},
  year      = {2022}
}
```
