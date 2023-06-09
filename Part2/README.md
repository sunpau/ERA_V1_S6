# S6 Assignment

The objective of the assignment is to achieve a validation accuracy of 99.4% on MNIST dataset
-  with less than 20k parameters 
-  within 20 epochs.

# Data
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains the following files:
  -  train-images-idx3-ubyte: training set images
  -  train-labels-idx1-ubyte: training set labels
  -  t10k-images-idx3-ubyte:  test set images
  -  t10k-labels-idx1-ubyte:  test set labels

The training set contains 60000 examples, and the test set 10000 examples.

In dataset loader, when train is set to True, train-images-idx3-ubyte is loaded. When train is set to False, t10k-images-idx3-ubyte is loaded.

Image Normalization is done while mean = 0.1307 and standard deviation = 0.3081

# Summary of the model
![Summary](https://github.com/sunpau/ERA_V1_S6/blob/main/images/Summary.png)
# Brief description
-  The input image size is 28x28x1. The number of layer and the number of channels are designed to reach RF of image size with the limitations of 20k parameters. The model contains              
    -  Input Layer (28x28x1) -->  Conv1(26x26x8)+ReLU - Conv2(24x24x16)+ReLU - Conv3(22x22x16)+ReLU - Conv4(20x20x32)+ReLU
    -  Transition Layer (MAX Pool(10x10x32) + 1x1 Convolution(10x10x8)) 
    -  Conv1(8x8x8)+ReLU - Conv2(6x6x16)+ReLU - Conv3(4x4x16)+ReLU - Conv4(2x2x32)+ReLU
    -   Global Average Pooling -> Softmax -> Output Layer
  ![architecture](https://github.com/sunpau/ERA_V1_S6/blob/main/images/Architecture.png)
-  Recepetive Field equal to the size of the image is preferred. However, MNIST dataset contains digits 0-9 and the border pixels does not contain any relevant information. So the last layer of the model has a receptive field of 26x26 which serves our purpose.  
-  All the kernals are 3x3, as using multiple 3x3 filters instead of using lesser larger kernals(5x5, 7x7 etc) will help achieve larger Receptive field with less computation. 
-  Transition Block - Only one Transition Block is used by using a MaxPooling followed by a 1x1 Convolutions. This is added after 4 layers
-  GAP - In the last layer, instead of Fully Connected layer, Global Average Pooling is used. This approach generates one feature map for each corresponding category of the classification task in the last Convolution layer.
-  Loss Function - nn.NLLLoss() is used as the loss function. It does not take probabilities but rather takes a tensor of log probabilities as input. Hence, in the last layer F.log_softmax() is used instead of just softmax function.
- Batch Size is taken as 256 with learning rate of 0.01 for first 15 epochs and then learning rate is reduced to 0.001 from 16th epochs. Higher batch size leads to faster convergence of the model as weights are updated after each propagation. However, depending on the input image, too large a batch size might not fit into the machine's memory.
