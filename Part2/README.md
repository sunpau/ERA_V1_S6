# S6 Assignment

The objective of the assignment is to achieve a validation accuracy of 99.4% on MNIST dataset
-  with less than 20k parameters 
-  within 20 epochs.

# Data
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains 60K images. 

# Summary of the model
# Brief description
-  The input image size is 28x28x1. The model contains              
  _**Input Layer (28x28x1)-> 4 Hidden Layers (Each with 3x3 kernals. Number of Channels 8->16->16->32)-> Transition Layer (MAX Pool + 1x1 Convolution) -> 4 Hidden Layer (Each with 3x3 kernals. Number of Channels 8->16->16->32) -> Global Average Pooling -> Softmax -> Output Layer**_
-  Recepetive Field equal to the size of the image is preferred. However, MNIST dataset contains digits 0-9 and the border pixels does not contain any relevant information. So the last layer of the model has a receptive field of 26x26 which serves our purpose.  
-  All the kernals are 3x3, as using multiple 3x3 filters instead of using lesser larger kernals(5x5, 7x7 etc) will help achieve larger Receptive field with less computation. 
-  Transition Block - Only one Transition Block is used by using a MaxPooling followed by a 1x1 Convolutions. This is added after 4 layers
-  GAP - In the last layer, FC is replaced with Global Average Pooling. This approach generates one feature map for each corresponding category of the classification task in the last Conv layer.
-  Loss Function - nn.NLLLoss() is used as the loss function. It does not take probabilities but rather takes a tensor of log probabilities as input. Hence, in the last layer F.log_softmax() is used instead of just softmax function.
- Batch Size is taken as 256 with learning rate of 0.01 for first 15 epochs and then learning rate is reduced to 0.001 from 16th epochs. Higher batch size leads to faster convergence of the model as weights are updated after each propagation. However, depending on the input image, too large a batch size might not fit into the machine's memory.