import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# 装载mnist数据集
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 卷积核大小
kernel_size1 = 3
kernel_size2 = 3


# 构建2层CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(    # 第一层卷积核
            nn.Conv2d(1, 32, kernel_size=kernel_size1, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(    # 第二层卷积核
            nn.Conv2d(32, 64, kernel_size=kernel_size2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 64, 10)     # 全连接层

    def forward(self, x):   # 前向函数
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # 将张量改变形状以便进入全连接层进行处理
        out = self.fc(out)
        return out


# 超参数
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 定义模型为cnn，loss计算标准为CrossEntropy和优化器
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

total_step = len(train_loader)  # 确定步数
for epoch in range(num_epochs): # 迭代
    for i, (images, labels) in enumerate(train_loader):
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 计算封闭测试精度和开发测试精度
with torch.no_grad():
    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0

    for images_train, labels_train in train_loader:
        outputs_train = cnn(images_train)
        _, predicted_train = torch.max(outputs_train.data, 1)
        total_train += labels_train.size(0)
        correct_train += (predicted_train == labels_train).sum().item()

    for images_test, labels_test in test_loader:
        outputs_test = cnn(images_test)
        _, predicted_test = torch.max(outputs_test.data, 1)
        total_test += labels_test.size(0)
        correct_test += (predicted_test == labels_test).sum().item()

    print('封闭测试精度: {:.2f}%'.format(100 * correct_train / total_train))
    print('开放测试精度: {:.2f}%'.format(100 * correct_test / total_test))
