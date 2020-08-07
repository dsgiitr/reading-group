# iSeeBetter & Other Super Resolution methods
## Paper
### [iSeeBetter: Spatio-temporal video super-resolution using recurrent generative back-projection networks](https://arxiv.org/abs/2006.11161)
## Abstract
Recently, learning-based models have enhanced the performance of single-image super-resolution (SISR). However, applying SISR successively to each video frame leads to a lack of temporal coherency. Convolutional neural networks (CNNs) outperform traditional approaches in terms of image quality metrics such as peak signal to noise ratio (PSNR) and structural similarity (SSIM). However, generative adversarial networks (GANs) offer a competitive advantage by being able to mitigate the issue of a lack of finer texture details, usually seen with CNNs when super-resolving at large upscaling factors. We present iSeeBetter, a novel GAN-based spatio-temporal approach to video super-resolution (VSR) that renders temporally consistent super-resolution videos. iSeeBetter extracts spatial and temporal information from the current and neighboring frames using the concept of recurrent back-projection networks as its generator. Furthermore, to improve the "naturality" of the super-resolved image while eliminating artifacts seen with traditional algorithms, we utilize the discriminator from super-resolution generative adversarial network (SRGAN). Although mean squared error (MSE) as a primary loss-minimization objective improves PSNR/SSIM, these metrics may not capture fine details in the image resulting in misrepresentation of perceptual quality. To address this, we use a four-fold (MSE, perceptual, adversarial, and total-variation (TV)) loss function. Our results demonstrate that iSeeBetter offers superior VSR fidelity and surpasses state-of-the-art performance.
# Supplementary Material
1. [Evolution of Image Super Resolution](https://towardsdatascience.com/an-evolution-in-single-image-super-resolution-using-deep-learning-66f0adfb2d6b)
2. [Collection of Single Image Super Resolution](https://github.com/YapengTian/Single-Image-Super-Resolution)
3. [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
4. [SRGAN](https://arxiv.org/abs/1609.04802)
5. [DBPN - Deep Back-Projection Networks For Super-Resolution]( https://arxiv.org/abs/1803.02735)
6. [DRLN - Densely Residual Laplacian Super-Resolution](https://paperswithcode.com/paper/densely-residual-laplacian-super-resolution)
7. [RCAN - Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)
8. [Multi Frame Bi-directional RNN](https://papers.nips.cc/paper/5778-bidirectional-recurrent-convolutional-networks-for-multi-frame-super-resolution.pdf)
9. [VSR-DUF - Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation](https://paperswithcode.com/paper/deep-video-super-resolution-network-using)
# Reference
```
@misc{chadha2020iseebetter,
    title={iSeeBetter: Spatio-temporal video super-resolution using recurrent generative back-projection networks},
    author={Aman Chadha and John Britto and M. Mani Roja},
    year={2020},
    eprint={2006.11161},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
