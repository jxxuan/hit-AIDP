from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from gensim.test.utils import datapath
import os
# from gensim.models import KeyedVectors
from nltk.corpus import stopwords

import logging

import jieba


# 下载BERT的预训练权重
model_name = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# 提取BERT模型的嵌入层参数
embedding_size = bert_model.config.hidden_size
embedding_weights = bert_model.embeddings.word_embeddings.weight


# class TextCNN(nn.Module):
#     def __init__(self, vec_dim, filter_num, sentence_max_size, label_size, kernel_size):
#         super(TextCNN, self).__init__()
#         chanel_num = 1
#         self.conv = nn.Sequential(
#             nn.Conv2d(chanel_num, filter_num, (kernel_size, vec_dim)),
#             nn.ReLU(),
#             nn.MaxPool2d((sentence_max_size - kernel_size + 1, 1))
#         )
#         self.fc = nn.Linear(filter_num, label_size)
#         self.dropout = nn.Dropout(0.5)
#         self.sm = nn.Softmax(0)
#
#     def forward(self, x):
#         in_size = x.size(0)
#         out = self.conv(x)
#         out = out.view(in_size, -1)
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out


class TextCNN(nn.Module):
    def __init__(self, num_labels):
        super(TextCNN, self).__init__()
        self.bert = bert_model
        self.bert_config = self.bert.config
        self.cnn = nn.Conv1d(
            in_channels=self.bert_config.hidden_size,
            out_channels=128,
            kernel_size=3
        )
        self.fc = nn.Linear(128, num_labels)
        self.dropout = nn.Dropout(0.5)


    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        # print(outputs.last_hidden_state.shape)
        pooled_output = torch.transpose(outputs[0], 1, 2)  # 使用pooled_output作为输入

        # pooled_output = pooled_output.unsqueeze(1)  # 增加一个维度，以适应Conv1d的输入形状 [batch_size, hidden_size, sequence_length]
        cnn_output = self.cnn(pooled_output)
        # print(cnn_output.shape)
        cnn_output = self.dropout(cnn_output)
        # print(cnn_output.shape)
        fc_input = torch.transpose(cnn_output, 1, 2)
        logits = self.fc(fc_input)
        return logits


class MyDataset(Dataset):

    def __init__(self, file_list, label_list, sentence_max_size, embedding, stopwords):
        self.x = file_list
        self.y = label_list
        self.sentence_max_size = sentence_max_size
        self.embedding = embedding
        self.stopwords = stopwords

    def __getitem__(self, index):
        # 读取文章内容
        words = []
        with open(self.x[index], "r", encoding="utf8") as file:
            for line in file.readlines():
                words.extend(segment(line.strip(), stopwords))
        sep = " "
        text = sep.join(words)
        encoded_input = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=300,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoded_input['input_ids'][0]
        # print(input_ids.shape)
        target = torch.tensor(self.y[index])
        target = F.one_hot(target, num_classes=2)  # 假设有2个类别
        target = target.squeeze(0)  # 去除第一维，使得形状变为 [num_classes]

        return input_ids, target

    def __len__(self):
        return len(self.x)


def load_stopwords(stopwords_dir):
    stopwords = []
    with open(stopwords_dir, "r", encoding="utf8") as file:
        for line in file.readlines():
            stopwords.append(line.strip())
    return stopwords


def segment(content, stopwords):
    res = []
    for word in jieba.cut(content):
        if word not in stopwords and word.strip() != "":
            res.append(word)
    return res


def get_file_list(source_dir):
    file_list = []  # 文件路径名列表
    # os.walk()遍历给定目录下的所有子目录，每个walk是三元组(root,dirs,files)
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    # 遍历所有文章
    if os.path.isdir(source_dir):
        for root, dirs, files in os.walk(source_dir):
            file = [os.path.join(root, filename) for filename in files]
            file_list.extend(file)
        return file_list
    else:
        print("the path is not existed")
        exit(0)


def get_label_list(file_list):
    # 提取出标签名
    label_name_list = [file.split("\\")[-2] for file in file_list]
    # 标签名对应的数字
    label_list = []
    for label_name in label_name_list:
        if label_name == "neg":
            label_list.append(0)
        elif label_name == "pos":
            label_list.append(1)
    return label_list


def generate_tensor(sentence, sentence_max_size, embedding):
    """
    对一篇文章生成对应的词向量矩阵
    :param sentence:一篇文章的分词列表
    :param sentence_max_size:认为设定的一篇文章的最大分词数量
    :param embedding:词向量对象
    :return:一篇文章的词向量矩阵
    """
    tensor = torch.zeros([sentence_max_size, embedding.embedding_dim])
    for index in range(0, sentence_max_size):
        if index >= len(sentence):
            break
        else:
            word = sentence[index]
            vector = embedding(torch.tensor(tokenizer.convert_tokens_to_ids(word)))
            tensor[index] = vector
    return tensor.unsqueeze(0)  # tensor是二维的，必须扩充为三维，否则会报错


def train_textcnn_model(model, train_loader, epoch, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 修改这里的net为model
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            # print("Batch Index:", batch_idx)
            # print("Data Shape:", data.shape)
            # print("Target Shape:", target.shape)
            # print(target)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            logging.info("train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / 64))

    print('Finished Training')


def textcnn_model_test(net, test_loader):
    net.eval()  # 必备，将模型设置为训练模式
    correct = 0
    total = 0
#    test_acc = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            logging.info("test batch_id=" + str(i))
            #data = data.to(cuda)
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Accuracy of the network on test set: %d %%' % (100 * correct / total))
            # test_acc += accuracy_score(torch.argmax(outputs.data, dim=1), label)
            # logging.info("test_acc=" + str(test_acc))


current_dir = os.getcwd()
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    train_dir = os.path.join(os.getcwd(), "../lab8/aclIdmb/train")  # 训练集路径
    test_dir = os.path.join(os.getcwd(), "../lab8/aclIdmb/test")  # 测试集路径
    stopwords_dir = os.path.join(os.getcwd(), "../lab8/stopwords.txt")  # 停用词
    # word2vec_dir = os.path.join(os.getcwd(),"glove.model.6B.50d.txt")  # 训练好的词向量文件,写成相对路径好像会报错
    net_dir = ".\\model\\net.pkl"
    sentence_max_size = 300  # 每篇文章的最大词数量
    batch_size = 64
    filter_num = 50  # 每种卷积核的个数
    epoch = 1  # 迭代次数
    kernel_size = 3  # 卷积核的大小
    label_size = 2
    lr = 0.001
    # 加载词向量模型
    logging.info("加载词向量模型")
    # 读取停用表
    stopwords = load_stopwords(stopwords_dir)
    # 加载词向量模型
    embedding_size = bert_model.config.hidden_size
    embedding_weights = bert_model.embeddings.word_embeddings.weight
    embedding = nn.Embedding.from_pretrained(embedding_weights)
    # 获取训练数据
    logging.info("获取训练数据")
    train_set = get_file_list(train_dir)
    train_label = get_label_list(train_set)
    train_dataset = MyDataset(train_set, train_label, sentence_max_size, embedding, stopwords)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 获取测试数据
    logging.info("获取测试数据")
    test_set = get_file_list(test_dir)
    test_label = get_label_list(test_set)
    test_dataset = MyDataset(test_set, test_label, sentence_max_size, embedding, stopwords)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # 定义模型
    net = TextCNN(num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    # 训练
    logging.info("开始训练模型")
    train_textcnn_model(net, train_dataloader, epoch, lr)
    # 保存模型
    torch.save(net, net_dir)
    logging.info("开始测试模型")
    textcnn_model_test(net, test_dataloader)
