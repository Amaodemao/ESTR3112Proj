import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net = Net()

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    net = Net()
    PATH1 = './cifar_net_SGD.pth'
    net.load_state_dict(torch.load(PATH1))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Prediction from SGD: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))

    PATH2 = './cifar_net_SGD_momentum.pth'
    net.load_state_dict(torch.load(PATH2))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Prediction from SGD with momentum: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    
    PATH3 = './cifar_net_Adagrad.pth'
    net.load_state_dict(torch.load(PATH3))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Prediction from Adagrad: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    
    PATH4 = './cifar_net_RMSprop.pth'
    net.load_state_dict(torch.load(PATH4))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Prediction from RMSprop: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    
    PATH5 = './cifar_net_Adam.pth'
    net.load_state_dict(torch.load(PATH5))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Prediction from Adam: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
                                  