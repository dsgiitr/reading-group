
# UNCERTAINTIES and BAYESIAN DL
## Paper
### [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
## Abstract
There are two major types of uncertainty one can model. Aleatoric uncertainty captures noise inherent in the observations. On the other hand, epistemic uncertainty accounts for uncertainty in the model -- uncertainty which can be explained away given enough data. Traditionally it has been difficult to model epistemic uncertainty in computer vision, but with new Bayesian deep learning tools this is now possible. We study the benefits of modeling epistemic vs. aleatoric uncertainty in Bayesian deep learning models for vision tasks. For this we present a Bayesian deep learning framework combining input-dependent aleatoric uncertainty together with epistemic uncertainty. We study models under the framework with per-pixel semantic segmentation and depth regression tasks. Further, our explicit uncertainty formulation leads to new loss functions for these tasks, which can be interpreted as learned attenuation. This makes the loss more robust to noisy data, also giving new state-of-the-art results on segmentation and depth regression benchmarks.

# Supplementary Material
1. [Uncertainties](https://towardsdatascience.com/building-a-bayesian-deep-learning-classifier-ece1845bc09)
2. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
3. [Variational Inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
4. [TFP Regressions](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html)
5. [Bayesian inference](https://towardsdatascience.com/probability-concepts-explained-bayesian-inference-for-parameter-estimation-90e8930e5348)
6. [Monte Carlo Methods](https://towardsdatascience.com/an-overview-of-monte-carlo-methods-675384eb1694)
# Reference
```
@misc{kendall2017uncertainties,
    title={What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?},
    author={Alex Kendall and Yarin Gal},
    year={2017},
    eprint={1703.04977},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

