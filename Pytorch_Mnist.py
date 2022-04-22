# Imports
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
import matplotlib.pyplot as plt

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Training function
def train(epoch):
    
    model.train()
    
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # adam step
        optimizer.step()
        running_loss += loss.item()
        _, predicted = scores.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        
    train_loss = running_loss / len(train_loader)
    train_acc = 100.*correct/total
    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    
    print('Epoch: ', epoch, ': train loss: %.4f' %train_loss, ' | train accuracy:%.4f' %train_acc)

# Testing function
def test(epoch):
    
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            running_loss += loss.item()
            
            _, predicted = scores.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
    test_loss = running_loss / len(test_loader)
    test_acc = 100.*correct/total
    test_losses.append(test_loss)
    test_accuracy.append(test_acc)
    
    print('| test loss: %.4f' %test_loss, ' | test accuracy:%.4f' %test_acc)

# Plot accuracy on the training and validation datasets 
def plot_accu():
    plt.plot(train_accuracy,'-o')
    plt.plot(test_accuracy,'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.savefig("accuracy.png")
    plt.clf()
    plt.close()

# Plot loss on the training and validation datasets
def plot_loss():
    plt.plot(train_losses,'-o')
    plt.plot(test_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig("loss.png")
    plt.clf()
    plt.close()



# Print pytorch cuda cudnn version
print("Pytorch Version : " + torch.__version__)
print("Cuda Version : " + torch.version.cuda)
print("Cudnn Version : " + str(torch.backends.cudnn.version()))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.get_device_name(0))

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load Data
train_dataset = datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists for recording metrics on each epoch
train_accuracy = []
train_losses = []
test_accuracy = []
test_losses = []

# Start training
for epoch in range(1,num_epochs+1): 
    train(epoch)
    test(epoch)
print('Finished Training')

# Plotting metrics
plot_accu()
plot_loss()



