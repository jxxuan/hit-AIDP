from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_

batchNumber = 64
epochNumber = 100
hiddenDim = 30


class CSVDataset(Dataset):
    # 加载数据集
    def __init__(self, path):
        # 将csv文件读取为dataframe
        df = read_csv(path, header=None)
        # 存储输入与输出，每行末尾列为因变量，其余为自变量
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # 输入为浮点型
        self.X = self.X.astype('float32')
        # 标记编码，确保因变量为浮点数
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # 获取数据集行数
    def __len__(self):
        return len(self.X)

    # 通过索引获取行
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # 获取训练、测试集行的索引
    def get_splits(self, n_test=0.3):
        # 根据比例确定训练集和测试集大小
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # 计算分裂
        return random_split(self, [train_size, test_size])


def prepare_data(path):
    # 加载数据集
    dataset = CSVDataset(path)
    # 划分训练集和测试集
    train, test = dataset.get_splits()
    # 准备数据加载器
    train_dl = DataLoader(train, batch_size=batchNumber, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# class MLP(Module):
#     # 定义模型元素
#     def __init__(self, n_inputs):
#         super(MLP, self).__init__()
#         # 输入层→隐藏层
#         self.hidden = Linear(n_inputs, hiddenDim)
#         kaiming_uniform_(self.hidden.weight, nonlinearity='relu')  # 初始化
#         self.act = ReLU()
#         # 隐藏层→输出层
#         self.output = Linear(hiddenDim, 1)
#         xavier_uniform_(self.output.weight)  # 初始化
#         self.act2 = Sigmoid()
#
#     # 前向传递
#     def forward(self, X):
#         # 输入层→隐藏层
#         X = self.hidden(X)
#         X = self.act(X)
#         # 隐藏层→输出层
#         X = self.output(X)
#         X = self.act2(X)
#         return X

class MLP(Module):
    # 定义模型元素
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # 输入层→隐藏层1
        self.hidden1 = Linear(n_inputs, hiddenDim)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')  # 初始化
        self.act = ReLU()
        # 隐藏层1→隐藏层2
        self.hidden2 = Linear(hiddenDim, hiddenDim)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 隐藏层2→输出层
        self.output = Linear(hiddenDim, 1)
        xavier_uniform_(self.output.weight)  # 初始化
        self.act3 = Sigmoid()

    # 前向传递
    def forward(self, X):
        # 输入层→隐藏层
        X = self.hidden1(X)
        X = self.act(X)
        # 隐藏层1→隐藏层2
        X = self.hidden2(X)
        X = self.act2(X)
        # 隐藏层→输出层
        X = self.output(X)
        X = self.act3(X)
        return X

def train_model(train_dl, model):
    # 定义优化
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 枚举epochs
    for epoch in range(epochNumber):
        # 枚举mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # 清除梯度
            optimizer.zero_grad()
            # 计算模型输出
            yhat = model(inputs)
            # 计算loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))
            # 更新模型权重
            optimizer.step()


def evaluate_model(test_dl, model):
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # 用测试集评估模型
        yhat = model(inputs)
        # 检索numpy数组
        yhat = yhat.detach().numpy()

        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # 四舍五入到类值
        yhat = yhat.round()
        # 存储
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # 计算精度
    acc = accuracy_score(actuals, predictions)
    return acc


def predict(row, model):
    # 将行转换为数据
    row = Tensor([row])
    # 做出预测
    yhat = model(row)
    # 检索numpy数组
    yhat = yhat.detach().numpy()
    return yhat


# 定义文件路径，获取训练集，测试集
path = './ionosphere/ionosphere.data'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))

# 定义模型
model = MLP(34)
print(model)

# 定义优化器，loss函数并用训练集训练模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lossfunc = torch.nn.NLLLoss().cuda()
train_model(train_dl, model)

# 评估模型，输出精度
acc = evaluate_model(train_dl, model)
print('Close Accuracy: %.3f' % acc)
acc = evaluate_model(test_dl, model)
print('Open Accuracy: %.3f' % acc)

# 用模型进行预测
row = [1, 0, 0.99539, -0.05889, 0.85243, 0.02306, 0.83398, -0.37708, 1, 0.03760,
       0.85243, -0.17755, 0.59755, -0.44945,
       0.60536, -0.38223, 0.84356, -0.38542, 0.58212, -0.32192, 0.56971, -0.29674,
       0.36946, -0.47357, 0.56811, -0.51171,
       0.41078, -0.46168, 0.21266, -0.34090, 0.42267, -0.54487, 0.18641, -0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
