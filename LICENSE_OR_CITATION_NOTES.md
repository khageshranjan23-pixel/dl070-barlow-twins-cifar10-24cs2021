# License and Citation Notes

## Academic Coursework License & Code Scope

This codebase and accompanying documentation were developed for:
- **Author:** Khagesh Ranjan (Roll: 24cs2021)
- **Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)
- **Course:** DL-070 Deep Learning Laboratory
- **Assignment:** Paper Reproduction & Ablation Study — Barlow Twins (Zbontar et al., ICML 2021)
- **Date:** October 2026
- **License:** MIT License (Applies strictly to original student-authored implementation code, scripts, test harness, documentation, and evaluation pipeline).

### Upstream Attribution & Third-Party Code
- **Barlow Twins Objective & Loss Formulation:** Conceptually attributed to Facebook Research (`facebookresearch/barlowtwins`, Zbontar et al., ICML 2021).
- **ResNet-18 Backbone Architecture:** Derived and adapted from PyTorch `torchvision.models` (BSD-3-Clause License) with modifications for $32 \times 32$ CIFAR-10 images (replacing the $7 \times 7$ conv1 with $3 \times 3$ stride 1 and replacing the initial max-pooling with identity).
- **CIFAR-10 Dataset:** Krizhevsky & Hinton (2009).

```
MIT License

Copyright (c) 2026 Khagesh Ranjan, Rajiv Gandhi Institute of Petroleum Technology (RGIPT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citations & Academic Attribution

If utilizing this codebase or experimental protocol, please cite the primary literature:

### 1. Primary Paper (Barlow Twins)
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

### 2. Primary Benchmark Dataset (CIFAR-10)
```bibtex
@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  journal={Master's thesis, Department of Computer Science, University of Toronto},
  year={2009}
}
```

### 3. Foundational Biological Principle (Barlow's Redundancy Reduction)
```bibtex
@article{barlow1961possible,
  title={Possible principles underlying the transformations of sensory messages},
  author={Barlow, Horace B},
  journal={Sensory Communication},
  volume={1},
  pages={217--234},
  year={1961}
}
```
