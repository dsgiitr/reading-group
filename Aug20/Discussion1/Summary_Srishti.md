# DEEP RESIDUAL LEARNING FOR IMAGE RECOGNITION
## Introduction
It is often affirmed that _“the deeper the better”_ when it comes to convolutional neural networks. This makes sense, since deeper models should be more capable to learn complex functions, owning to a larger set of parameters. However, problems have been noted to arise while optimizing deep networks.
### 1.	Vanishing/Exploding Gradients: 
Vanishing Gradients hamper convergence and considerably affect the model’s capacity to converge efficiently. This problem, however, can be rectified by using normalized initialization and intermediate normalization layers.
###  2.	Degradation Problem:
As the network depth increases, the accuracy falls into saturation (which is not unexpected), and then decreases. Surprisingly, such a drop is not caused by overfitting, since it is observed that adding more layers to a network with a suitable depth results in not just a higher test error, but a higher training error too. This problem indicates that not all systems are similarly easy to optimize. This was one of the bottlenecks of VGG. They couldn’t go as deep as wanted, because they started to lose generalization capability.
Residual networks propose a solution to these problems by applying residual learning as explained below.



## RESIDUAL LEARNING
**Skip Connections**: Once the network tends to reach saturation accuracy, the task at hand is to learn identity mappings. The degradation problem suggests that solvers might have difficulty in approximating identity mapping by multiple non-linear layers. If an identity mapping were optimal, it would be easier of the non-linear layers to push the residual to zero than to fit an identity mapping. 
![image of skipconnection](https://miro.medium.com/max/463/1*tEaVn-9OEPCre0lmHAJWyw.png)
 
Let the desired initial mapping be denoted by H(x).
By adding the value of the input(x), directly to the output, we let the stacked non-linear layers fit another mapping of F(x) := H(x)-x.
The output can now be written as 
######         H(x) := F(x) + x
The operation F(x)+x is performed by a shortcut connection and element wise addition, after which the nonlinearity is implemented.
Now the goal of the stack of layers becomes to approximate the residual result tot zero. Since it is easier to optimize the residual mapping than the original mapping, shortcut connections provide easier implementation without adding extra parameters or increasing computational complexity, and the entire network can still be trained end-to-end by Stochastic Gradient Descent with backpropagation.
Here, the stack of layers must contain 2 or more layers as if there is only a single layer, the equation becomes similar to a linear layer, which does not contribute to the network. 
######  y= W1x+x 
To impose a ‘skip connection, the dimensions of the output should be same as the input. If that is not the case, the paper proposes the following solutions:
1.	Using zero padding to increasing dimensions
2.	Performing a linear projection to the input (Ws) to match the dimensions.
                                              y=F(x,{Wi} )+ Wsx  
### Solving the problem of vanishing gradients:
Using skip connections also solves the problem of vanishing gradients. This is because when the network is too deep, the gradients from where the loss function is calculated easily shrink to zero after several applications of the chain rule. This result on the weights never updating its values and therefore, no learning is being performed.
With skip connections, the gradients can flow directly through the skip connections backwards from later layers to initial filters.




## Network Architecture:
![network architecture](https://i.stack.imgur.com/xuzKK.png)

The right side of the figure above shows the residual network structure diagram. Some of the constant mapping lines in the figure are solid lines, and some of the connecting lines are dotted lines. The connections having different dimensions are indicated by dotted lines. The connection dimensions of the solid line are the same. These two different kinds of skip connections are so called in the paper as __Identity Shortcut__ and __Projection Shortcut__. The identity shortcut is done simply bypassing the input volume to the addition operator. The projection shortcut performs a convolution operation to ensure the volumes at this addition operation are the same size. From the paper we can see that there are 2 options for matching the output size. Either padding the input volume or performing 1x1 convolutions. 
The convolutional layers used mainly apply a 3×3 filter and follow two principles:
(1)  For the same output feature map size, these layers have the same number of filters.
(2)  If the feature size is reduced by half, the number of filters is doubled to maintain the time complexity of each layer.
 
 
The down sampling of the volume though the network is achieved by increasing the stride to 2 instead of a pooling operation like normally CNNs do. In fact, only one max pooling operation is performed in our Conv1 layer, and one average pooling layer at the end of the ResNet, right before the fully connected dense layer




### Bottleneck Architectures: 
![bottleneck pic](https://miro.medium.com/max/766/1*zS2ChIMwAqC5DQbL5yD9iQ.png)
 
Bottleneck Architectures were introduced in the paper to implement more efficient skip connections while training deeper networks. 
As shown in the figure above, the two-layer structure is changed into three layers, and the 3×3, 3×3 structures are replaced by a 1×1, 3×3, and 1×1 convolution structure, and the functions they implement are exactly the same. But the advantage is that the amount of parameters is reduced。The first 1×1 function is dimension reduction, so that the 3×3 operation has a lower input/output dimension, and the second 1×1 role is to restore the dimensions to the initial dimensions. The parameter amount of the initial residual block is 3 × 3 × 256 × 256 × 2 = 1179648, and the module parameter amount using the bottleneck structure is 1 × 1 × 256 × 64 + 3 × 3 × 64 × 64 + 1 × 1 × 64 × 256 = 69632, which shows that the amount of parameters is reduced by 16.94 times. Therefore, in the conventional Resnet or shallower network structure, the residual block on the left can be used. But in deeper networks (Resnet-101), using the bottleneck structure on the right can reduce network parameters. For deeper networks like ResNet50, ResNet152, etc, bottleneck design is used. 


### Results:
![result pic](https://miro.medium.com/max/753/1*-_ED04HNCNz7HFyqQtbTtg.png)

The advantages of resnets over plain networks can easily be seen in the following figure where the error rates for plain networks increase on increasing number of layers and decrease in case of resnets.
The authors of the paper also experimented with implementing architectures with greater than 1000 layers, but deep networks could not overcome overfitting, leading to a greater test set error, even though the training error did not increase.




