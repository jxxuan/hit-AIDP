import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 加载数据集
df = pd.read_csv('ionosphere.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 将类别标签转换为数值形式
y = np.where(y == 'g', 1, 0)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# 创建训练数据集对象
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# 设置模型超参数
input_dim1 = X_train.shape[1]
hidden_dim1 = 30
output_dim1 = 1
batch_size = 20
num_epochs = 100

# 初始化模型
model = MLP(input_dim1, hidden_dim1, output_dim1)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# 将测试集数据转换为PyTorch张量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 在测试集上进行预测
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = torch.round(outputs)

with torch.no_grad():
    outputs_train = model(X_train_tensor)
    predicted_train = torch.round(outputs_train)


# 计算准确率
accuracy = (predicted == y_test_tensor.view(-1, 1)).sum().item() / len(y_test_tensor)
print(f'Open Test Accuracy: {accuracy:.4f}')

accuracy_train = (predicted_train == y_train_tensor.view(-1, 1)).sum().item() / len(y_train_tensor)
print(f'Close Test Accuracy: {accuracy_train:.4f}')
