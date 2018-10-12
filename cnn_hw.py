import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_transform = transforms.ToTensor()
train_data = MNIST(root='./data',train=True,download=True,transform=data_transform)
test_data = MNIST(root='./data',train=False,download=True,transform=data_transform)
# print(len(train_data),len(test_data))
batch_size = 20
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)
classes = ['0','1','2','3','4','5','6','7','8','9']

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
fig = plt.figure(figsize=(25,4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2,batch_size/2 , idx+1, xticks = [],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmap='gray')
    # print(np.squeeze(images[idx]).shape) # Size = 28x28
    ax.set_title(classes[labels[idx]])
#plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 1 input,10 output channel/featuremap, 3x3 convolution kernel, output size = (W-F)/S +1 = (28-1)/1 + 1 = 28 
        self.conv1 = nn.Conv2d(1,32,3) # output size is (32,28,28)
        self.pool = nn.MaxPool2d(2,2)  # output size is (32,14,14)
        self.conv21 = nn.Conv2d(32,64,3) # input 14, output size = (14-1)/1 + 1 = 14, ie., (64,14,14)
        self.conv22 = nn.Conv2d(32,64,3) # input 14, output size = (14-1)/1 + 1 = 14, ie., (64,14,14)
        self.conv31 = nn.Conv2d(64,512,3) # input 7, output size = (7-1)/1 + 1 = 7, ie., (512,7,7)
        self.conv32 = nn.Conv2d(64,512,3) # input 7, output size = (7-1)/1 + 1 = 7, ie., (512,7,7)
        self.fc1 = nn.Linear(1024, 1000) #(input size, output size)
        self.fc2 = nn.Linear(1000, 500) #(input size, output size)
        self.fc3 = nn.Linear(500, 10) #(input size, output size)

    def forward(self,x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x21 = self.pool(F.relu(self.conv21(x1)))
        x22 = self.pool(F.relu(self.conv22(x1)))
        x31 = self.pool(F.relu(self.conv31(x21)))
        x32 = self.pool(F.relu(self.conv32(x22)))
        combined = torch.cat((x31.view(x31.size(0),-1),x32.view(x32.size(0),-1)),dim=1)
        # print(x31.size())
        # print(x32.size())
        # print(combined.size())
        # x = x.view(x.size(0),-1) # flatten inputs to a vector
        xfc1 = F.relu(self.fc1(combined))
        xfc2 = F.relu(self.fc2(xfc1))
        xfc3 = F.relu(self.fc3(xfc2))
        x = F.log_softmax(xfc3,dim=1) # converts 10 outputs to distribution of class scores
        return x

net = Net()
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001)

correct = 0
total = 0

for images,labels in test_loader:
    outputs = net(images)
    _, predicted = torch.max(outputs.data,1) # gets maximum value in output-list of predicted class scores
    total += labels.size(0)
    correct += (predicted == labels).sum()

accuracy = 100.0 * correct.item() / total
print('Accuracy before training:', accuracy)

def train(n_epochs):
    loss_over_time = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            inputs, labels = data # get input images and corresponding labels
            optimizer.zero_grad() # set parameter weight gradients as zero
            outputs = net(inputs) # forward pass to get outputs
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # backward pass to calculate parameter gradients
            optimizer.step() # update the parameters
            running_loss += loss.item()

            if batch_i % 1000 == 999:
                avg_loss = running_loss/1000
                loss_over_time.append(avg_loss)
                print('Epoch {}, Batch {}, Avg. Loss: {}'.format(epoch+1,batch_i+1,avg_loss))
                running_loss = 0.0
    print('Finished training')
    return loss_over_time

n_epochs = 30
training_loss = train(n_epochs)

plt.plot(training_loss)
plt.xlabel('1000s of batches')
plt.ylabel('loss')
plt.ylim(0,2.5)
#plt.show()

test_loss = torch.zeros(1)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
net.eval() # set module to evaluation mode

for batch_i,data in enumerate(test_loader):
    inputs, labels = data
    outputs = net(inputs)
    loss = criterion(outputs,labels)
    test_loss = test_loss + ((torch.ones(1)/(batch_i+1))*(loss.data - test_loss))
    _, predicted = torch.max(outputs.data,1)

    correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
    for i in range(batch_size):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

print('Test loss: {:.5f}\n'.format(test_loss.numpy()[0]))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (classes[i],100*class_correct[i]/class_total[i],np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\n Test Accuracy of overall: %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct),np.sum(class_total)))

dataiter = iter(test_loader)
images, labels = dataiter.next()

preds = np.squeeze(net(images).data.max(1,keepdim=True)[1].numpy())
images = images.numpy()

fig = plt.figure(figsize=(25,4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2,idx+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmap='gray')
    ax.set_title("{} {}".format(classes[preds[idx]],classes[labels[idx]], color=("green" if preds[idx] == labels[idx] else "red")))

model_dir = r'C:\Git\i_cnn_hw'
model_name = r'handwriting.pt'
torch.save(net.state_dict(),model_dir+model_name)