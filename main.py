import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Define a simple convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

net = Net()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Optimizers
optimizers = {
    'SGD': optim.SGD(net.parameters(), lr=0.001),
    'SGD_momentum': optim.SGD(net.parameters(), lr=0.001, momentum=0.9),
    'Adam': optim.Adam(net.parameters(), lr=0.001),
    'Adagrad': optim.Adagrad(net.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(net.parameters(), lr=0.001)
}

# Function to train and evaluate the model
def train_and_evaluate(optimizer_name, optimizer, epochs=2):
    train_loss = []
    test_accuracy = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0

        # Evaluate on test data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracy.append(accuracy)
        print(f'Accuracy of the network on the 10000 test images: {accuracy} %')

    PATH = f'./cifar_net_{optimizer_name}.pth'
    torch.save(net.state_dict(), PATH)
    return train_loss, test_accuracy

if __name__ == '__main__':
    # Training and evaluation
    results = {}
    for optimizer_name, optimizer in optimizers.items():
        initialize_weights(net)
        print(f"Training with {optimizer_name}")
        train_loss, test_accuracy = train_and_evaluate(optimizer_name, optimizer)
        results[optimizer_name] = {'train_loss': train_loss, 'test_accuracy': test_accuracy}



    # Plotting the results
    for optimizer_name, result in results.items():
        plt.plot(result['test_accuracy'], label=f'{optimizer_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Optimizer Comparison on CIFAR-10')
    plt.legend()
    plt.show()  