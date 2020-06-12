# Fractional Max Pooling

## INTRODUCTION
* [Link To Paper](https://arxiv.org/pdf/1412.6071.pdf)
* Here is the [implementation](https://github.com/diogo149/theano_fractional_max_pooling) in theano.
* Covolutional Networks have been evolved overtime. Many researchers had put their efforts in designing Different kinds of Activation Layers, Different Size of Convolution Layers, reducing overfitting by Dropout, BatchNormalization.
* However, very little thought has been putup into updating traditional MaxPooling Layers.
* Pooling Layers are building blocks of the CNN
* It reduces spatial dimension of the data. Like if we have *N<sub>in</sub>* x *N<sub>in</sub>* Matrix, and we apply MaxPool Layer on that it spatial dimensions will reduce to *N<sub>out</sub>* x *N<sub>out</sub>*. Where reduction factor *α* = N<sub>in</sub> / N<sub>out</sub>

## Traditional MaxPool Layer

* Traditionally 2 x 2 MaxPool layer has been used for Spatial Pooling

### Advantages
* Fast, reduces size of hidden layer quickly.
* Encodes degree of invariance with respect to translations and elastic distortions.

### Disadvantages
* Disjoint nature of Pooling operations can reduce generalization.
* MaxPooling reduces size so quickly that to build a deep network stack of Convolution Layers are required.

## Alternatives Proposed Before
* Using 3 x 3 pooling regions overlapping with stride 2 as used in [AlexNet](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf)
* [Stochastic Pooling](https://arxiv.org/pdf/1301.3557.pdf)

## Fractional Max Pooling
* Reduces spatial size of Image by a factor of α, where 1 < α < 2
* Introduce randomness like Stochastic Pooling
* Overlapping pooling regions

### How to design it?
* Input : *N<sub>in</sub>* x *N<sub>in</sub>*, Output : *N<sub>out</sub>* x *N<sub>out</sub>*, reduction factor *α* = N<sub>in</sub> / N<sub>out</sub>
* General Idea if to divide N<sub>in</sub> x N<sub>in</sub> square into N<sub>out</sub>^2 pooling regions *(P<sub>i,j</sub>)*
* Output<sub>i,j</sub> = max<sub>(k,l) ∈ P <sub>i,j</sub></sub> Input<sub>k,l</sub>
* To do this, generate two increasing subsequences *(a<sub>i</sub>)* and *(b<sub>i</sub>)*, 0 <= i <= N<sub>out</sub>, starting with 1 and ending at N<sub>in</sub> and with an icrement of 1 or 2.
* Now, we can generate two kind of Pooling regions
* Disjoint                           | Overlapping
  -----------------------------------|-------------
  P = [a<sub>i-1</sub>, a<sub>i</sub>-1] x [b<sub>j-1</sub>, b<sub>j</sub> -1] | P = [a<sub>i-1</sub>, a<sub>i</sub>] x [b<sub>j-1</sub>, b<sub>j</sub>]
* To generate integer sequence two different approaches
  * *random* = increments are obtained by random permutations of appropriate number of 1 and 2
  * *pseudorandom* = increments are obtained by a<sub>i</sub> = ceiling(α(i+u)), with *α* ∈ (1,2) and *u* ∈ (0,1)
* While training or testing, whenever CNN with FMP is applied on the dataset, we can generate different integer sequences and then average it to generate ensemble of it.

### Which limitations does it overcome over traditional MP?
* *Disjoint* as well as *Overlapping* Pooling Regions.
* *Randomness* included like Stochastic Pooling
* Reduction factor *α* reduced to *α* ∈ (1,2), so now we can generate deep network without 

## Key Points Observations
* *Random* Fractional Max Pooling may undefit when combined with DropOut.
* Improvement over traditional MP is substantial.
* *Overlapping* FMP better than *Disjoint* FMP

## Notable Results
* Paper was released in 2015. Still, [best results on CIFAR10](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).
* Near the best results on CIFAR100 and MNIST.

## Further possible improvement 
* ![Fractional Max Pooling](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/55dda8f230566867acbfaa7bdd08fd8c7b8721ed/2-Figure2-1.png)
* Looking at the distortions, it is decomposible in both *x* and *y* directions. Can we explore pooling regions which are different than the regions given by equation above?
