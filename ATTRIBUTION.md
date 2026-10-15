# Attribution and References

This repository contains an independent reproduction and ablation study of Barlow Twins on CIFAR-10.

## 1. Primary Paper Citation
The Barlow Twins objective and self-supervised learning paradigm are proposed and authored by:
```bibtex
@inproceedings{zbontar2021barlow,
  title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={12310--12320},
  year={2021},
  organization={PMLR}
}
```

## 2. Reference Implementation
- **Official Repository:** [`facebookresearch/barlowtwins`](https://github.com/facebookresearch/barlowtwins)
- **License:** MIT License (Copyright (c) Meta Platforms, Inc. and affiliates)

## 3. ResNet-18 Backbone Attribution
- **Source:** PyTorch Torchvision (`torchvision.models.resnet18`)
- **License:** BSD 3-Clause License (Copyright (c) Soumith Chintala 2016)
- **Adaptation:** Adapted for $32 \times 32$ CIFAR-10 images (replacing the $7 \times 7$ stride-2 convolution with a $3 \times 3$ stride-1 convolution, replacing the initial max-pooling with an identity mapping, and replacing the final classification head with an identity mapping to output 512-dimensional representations).

## 4. Benchmark Dataset Citation
```bibtex
@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  journal={Master's thesis, Department of Computer Science, University of Toronto},
  year={2009}
}
```
