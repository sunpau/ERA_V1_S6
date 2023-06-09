from torchvision import transforms
import matplotlib.pyplot as plt

"""CODE BLOCK: 6"""
def plot_data(batch_data, batch_label, num_plots, row, col): 

# batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(num_plots):
        plt.subplot(row,col,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

"""CODE BLOCK: 11"""
def plot_loss_accuracy(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
	
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
	
# def model_summary(model):
# 	# !pip install torchsummary
# 	# from torchsummary import summary
# 	# use_cuda = torch.cuda.is_available()
# 	# device = torch.device("cuda" if use_cuda else "cpu")
# 	# model = Net().to(device)
# 	summary(model, input_size=(1, 28, 28))