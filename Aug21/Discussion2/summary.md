# Visualizing and Understanding Convolutional Networks

## Purpose:

This paper introduces visualization techniques which gives insight into
the function of intermediate feature layer and helps us to understand
the performance contributions from various model layers. This will
further enable us to observe the evolution of features during training
and to diagnose problems with the model.

## Problem:

Before this paper was published, visualizations were limited to the 1st
layer\'s filters. This was because we were taking a direct convolution
between the weights of the first layer and the input pixels, hence those
filters could be projected to the pixel space. But this interpretation
was not applicable to the weights of the hidden layers. Below are the
filters learned by the first layer of AlexNet (Why do visualizing these
filters give what they are looking for? This comes from the fact that
when a filter is made to convolve with a piece of data, the output is
maximum when that piece of data is the filter itself). If this is done
to the hidden layers, the results are not satisfying or interpretable.

![](.\media\image1.png)

## Approach:

What we need is a way to map these not so interpretable filters back to
the input pixel space. This is done using a Deconvolutional Network
(**deconvnet**). To find which part of the image activates a given
convnet activation, we set all the other activations in the layer to 0
and pass the feature maps (activations) as input to the attached
deconvnet.

**Deconvolutional Network:** They were proposed as a way of performing
unsupervised learning. Here they are just used to examine an already
learned result. 
![](.\media\image2.png)

-   **Unpooling**: Pooling operation is
    irreversible; hence they try to generate an approximate result by
    keeping track of locations of maxima (for MaxPooling) within each
    pooling region in a set of 'switch' variables. Hence while
    unpooling, the structure of stimulus is preserved.

-   **Rectification**: Rectification is done using the ReLU
    non-linearity to ensure that the feature reconstructions are valid
    (positive).

-   **Filtering**: The convolution operation of the convnet is reversed
    by using the transposed version (flipping each filter vertically and
    horizontally) of the learned filters and applying them to the
    reconstructed signal from above.

These steps are repeated until we get to the input layer. The
reconstructions obtained from a single activation resembles a small
piece of the input image weighted according to their contribution
towards the feature activation.

## Results:

### Experiment 1: Feature Visualization

For each layer, they select the top 9 strongest activations and project
each to the pixel space. These projections reveal the different
structures that excite a given feature map. Alongside these projections,
the original image patches are shown too. Hierarchical nature of
features can be seen.

![](.\media\image3.png)*Layer 1*:
edges. *Layer 2*: corners and colour conjunctions. *Layer 3*: captures
similar textures. *Layer 4*: class-specific variation (dog faces, bird's
legs). *Layer 5*: entire objects with significant pose variation
(keyboards, dogs).

### Experiment 2: Feature Evolution during Training

For a given feature map, they take the strongest activation, project it
down to the pixel space (using the deconvnet approach) and study its
evolution over various epochs (1, 2, 5, 10, 20, 30, 40, 64).

We see that the lower layer features develop and converge within a few
epochs, whereas the higher layer develop only after a significant number
of epochs.![](.\media\image4.png)

### Experiment 3: Feature Invariance

Image augmentations (translation, rotation, scaling) have a dramatic
effect in the first layer of the model, whereas a lesser impact is seen
in the top feature layer. This is expected

### Experiment 4: Architecture Selection

They show that these projections of feature maps can help us improve a
particular architecture. They project the feature maps of a pretrained
Alex Net and study them to find 2 problems:

-   First layer filters are a mix of extremely high and low frequency
    information, with little coverage of the mid frequencies.

-   2nd layer visualization shows aliasing artifacts caused by the large
    stride 4 used in the 1st layer convolutions

To fix these problems, they make the following changes:

-   Reduced the 1st layer filter size from 11x11 to 7x7

-   Made the stride of the convolution 2 rather than 4

This improved the classification
accuracy![](.\media\image5.png)

### Experiment 5: Occlusion Sensitivity

Now we also want to confirm if our CNN is identifying the location of the
object in the image. For this, they covered regions of the image with a
grey box and performed the training. Below are the amazing results:
![](.\media\image6.png)
