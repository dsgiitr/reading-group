# Adversarial Autoencoders

## Abstarct
In this paper, we propose the “adversarial autoencoder” (AAE), which is a probabilistic autoencoder that uses the recently proposed generative adversarial networks
(GAN) to perform variational inference by matching the aggregated posterior of
the hidden code vector of the autoencoder with an arbitrary prior distribution.
Matching the aggregated posterior to the prior ensures that generating from any
part of prior space results in meaningful samples. As a result, the decoder of the
adversarial autoencoder learns a deep generative model that maps the imposed prior
to the data distribution. We show how the adversarial autoencoder can be used in
applications such as semi-supervised classification, disentangling style and content
of images, unsupervised clustering, dimensionality reduction and data visualization.
We performed experiments on MNIST, Street View House Numbers and Toronto
Face datasets and show that adversarial autoencoders achieve competitive results
in generative modeling and semi-supervised classification tasks.

# Reference
```
@misc{makhzani2015adversarial,
    title={Adversarial Autoencoders},
    author={Alireza Makhzani and Jonathon Shlens and Navdeep Jaitly and Ian Goodfellow and Brendan Frey},
    year={2015},
    eprint={1511.05644},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
