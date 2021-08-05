# Very Deep Convolutional Networks for Large-Scale Image Recognition

Karen Simonyan & Andrew Zisserman, **ICLR 2015**

## Summary

This paper investigates the effect of increasing depth of a convolutional network on the ImageNet Challenge- $2014$. Instead of using less number of large-sized filters, it suggests reducing the size of filters and increasing the depth of the network, thus achieving better performance on the image classification and segmentation tracks of the ImageNet Challenge. 

## Key Idea

The key idea behind this paper is that a deeper convolutional network having smaller filters has nearly the same number of weights as that of a shallow network with larger filters and receptive fields, while helping the network fit better to the training data. This can be illustrated better by considering an example of a stack of three $3 \times 3$ filters with stride $1$. We observe that the effective receptive field of this stack of three $3 \times 3$ filters is $7 \times 7$. Thus, we have incorporated $3$ non-linearities instead of $1$ (in case of a single $7 \times 7$ layer), while simultaneously decreasing the number of parameters required from $7^2 = 49$ per channel to $3 \times 3^2 = 27$ per channel. 

## Architecture 

The architecture of the VGG16 network (with $16$ deep layers) is shown in the diagram below. 


<img src='VGG16.png' width='600'>


## Important Observations

- It was conjectured that greater depth and smaller convolutional filters led to an inherent regularization being applied on the network. This, coupled with pre-initialization of certain layers allowed the network to converge in less number of epochs.
- This network performed exceptionally well in the localisation task of the ImageNet challenge.

## Drawbacks

- It was observed that this network was slow to train and deploy due to its depth and number of fully connected node.
- The weights involved in this network are by themselves quite large. 

## References
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf )
- [VGG16- Neurohive.io Blog](https://neurohive.io/en/popular-networks/vgg16/)



