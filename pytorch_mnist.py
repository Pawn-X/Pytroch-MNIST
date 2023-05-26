import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
"""
卷积运算 使用mnist数据集
"""
# Super parameter ------------------------------------------------------------------------------------
batch_size = 128
learning_rate = 0.0001
EPOCH = 60

#图像预处理,compose就是把很多步骤合在一起，Normaliz用均值和标准差归一化张量图像(output[channel] = (input[channel] - mean[channel]) / std[channel])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)  # train=True训练集，=False测试集
# 训练集乱序，测试集有序
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Design model using class ------------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

#construct net
model = Net()
if torch.cuda.is_available():
    model = model.cuda()

# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
if torch.cuda.is_available():
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train and Test class --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来
        running_loss += loss.item()
        #600个batch输出一次loss平均值
        if batch_idx % 600 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch+1, running_loss / 600))

def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            pred = outputs.argmax(dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            correct += (pred == labels).sum().item()
    acc = correct / len(test_loader.dataset)
    print('[%d / %d]: Accuracy on test set: %.2f %% ' % (epoch + 1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
