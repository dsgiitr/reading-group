# ImageNet Classification with Deep Convolutional Neural Networks

Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton, NIPS, 2012


### General Details

* **Approximate duration:** 25 minutes
* **Presented by:** Karan Desai
* **Prerequisites:** Reader must know how a simple feedforward neural networks are designed.


### Overview

This paper ...
* Belongs to Imagenet LSVRC 2012 winners (Top-1 error of 35% and Top-5 error of 17%).
* Introduces today's common Deep Convolution Neural Network architecture.
* Describes the complete architecture of **AlexNet** model - specifications of each layer
  in an 8 layer deep model.
* Discusses training specifications and data augmentation approaches.


### Dataset and Preprocessing

* 1.2 million images - 1 million train, 50k validation, 150k test images of variable resolutions and sizes.
* Resize smaller dimension to 256 pixels and crop central patch.
* Subtract mean image from all these images (pixelwise means across complete training dataset).


### AlexNet Architecture and Training

* 8 layers network (said to be, they consider `conv + pool` to be single layer) with 650,000 neurons.
* 5 `conv + pool` layers, 2 `fully connected` layers and a `softmax` layer with 1000 outputs for 1000 classes.
* Takes `224 x 224 x 3` image as input, all intermediate layers have `relu` activation.
* Parallel training on 2 GTX 580 GPUs (3 GB memory).
* Used Stochastic Gradient Descent with momentum, weight decay and learning rate annealing.


### Prevention of Overfitting

* Data Augmentation
    - Extracts random crops of (224 x 224) from an image and performs translations and / or flipping during training.
    - Alters RGB pixel values by performing PCA on training set, and adding multiples of eigenvalues times a random 
    variable drawn from a Gaussian to image. This provides invariance to changes in intensity and color of illumination.

* Dropout
    - Dropout prevents overfitting. Randomly drops half of the neurons in the fully connected layers, and can be 
    interpreted as averaging over exponentially-many dropout networks.

* Local Response Normalization
    - Divide activation output of each neuron by a term proportional to sum of squares of activations of all neurons of 
    a few neighbouring channels.

### Minutae

* ReLU activation preferred over tanh and encouraged to be used due to non saturating behaviour.
* Usage of dropout in a practical model showcased for the first time.
* Optimized CUDA code capable of running on parallel GPUs has been released (Bug deal in 2012 but not now in 2017).
* The paper states that current model is optimal because removing any one layer affects the accuracy in a bad way - 
design decisions are motivated solely by results.
