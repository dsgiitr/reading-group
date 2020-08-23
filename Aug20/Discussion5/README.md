
# Attention Is All You Need

## Paper

### [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

# Supplementary Material

1. [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
3. [An introduction to Attention](https://towardsdatascience.com/an-introduction-to-attention-transformers-and-bert-part-1-da0e838c7cda)
4. [Self Attention and Transformers](https://towardsdatascience.com/self-attention-and-transformers-882e9de5edda)
5. [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
6. [Attention Mechanism](https://blog.floydhub.com/attention-mechanism/)

# Reference
```
@inproceedings{46201,
title	= {Attention is All You Need},
author	= {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
year	= {2017},
URL	= {https://arxiv.org/pdf/1706.03762.pdf}
}
```