import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount
from tqdm import tqdm


#This class contains the architecture of the neural network
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
          
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 8, 3, bias=False), #Image Input: 28x28x1 -> 26x26x8  #Receptive Field 1 -> 3
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, 3, bias=False), #Input: 26x26x8 -> 24x24x16  #Receptive Field 3 -> 5
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 16, 3, bias=False), #Input: 24x24x16 -> 22x22x16  #Receptive Field  5-> 7
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, bias=False), #Input: 22x22x16 -> 20x20x32 #Receptive Field  7-> 9
        nn.ReLU(),
        nn.BatchNorm2d(32),
        #Transition Block = MaxPool + 1x1 Convolution
        nn.MaxPool2d(2, 2),    #Input: 20x20x32 -> 10x10x32  #Receptive Field  9 -> 10
        nn.Conv2d(32, 8, 1, bias=False),   #Input: 10x10x32 -> 10x10x8  #Receptive Field  10 -> 10
        nn.ReLU(),
        nn.Dropout(0.1)
    )
          
    self.conv2 = nn.Sequential(
        nn.Conv2d(8, 8, 3, bias=False),  #Input: 10x10x8 -> 10x10x8  #Receptive Field  10 -> 14
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, 3, bias=False),  #Input: 10x10x8 -> 10x10x16  #Receptive Field  14 -> 18
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 16, 3, bias=False),  #Input: 10x10x16 -> 10x10x16  #Receptive Field  18 -> 22
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, bias=False),  #Input: 10x10x16 -> 10x1032  #Receptive Field  22 -> 26
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout(0.1)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(32, 10, 1, bias=False),  #GAP implementation - 10 Classes to be predicted so 10 feature map is generated
        nn.AdaptiveAvgPool2d((1,1))  #Average is calculated for each Feature Map.
        
    )
          
          
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)

    x = x.view(-1, 10) 
    x = F.log_softmax(x, dim=1)
    return x


    
def train_model(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()  # zero the gradients- not to use perious gradients

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()   #updates the parameter - gradient descent
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  train_acc = 100*correct/processed
  train_loss = train_loss/len(train_loader)
  return train_acc, train_loss
  

def test_model(model, device, test_loader, criterion):
    model.eval() #set model in test (inference) mode

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc, test_loss
# def model_summary():
# 	!pip install torchsummary
# 	from torchsummary import summary
# 	use_cuda = torch.cuda.is_available()
# 	device = torch.device("cuda" if use_cuda else "cpu")
# 	model = Net().to(device)
# 	summary(model, input_size=(1, 28, 28))