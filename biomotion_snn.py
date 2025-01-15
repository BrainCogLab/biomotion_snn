import numpy as np
import scipy.io as sio
# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
from tqdm import *
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from torchsummary import summary
from sklearn.manifold import TSNE
import torch.nn.utils.prune as prune
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def l2_regularization(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2
    return l2_reg

def generate2int(start, end):
    if start >= end:
        raise ValueError("start 必须小于 end")

    first = random.randint(start, end)

    # 确保第二个随机数与第一个不相同
    second = first
    while second == first:
        second = random.randint(start, end)

    return first, second

def ndarry2tensor(x):
    x = torch.tensor(x, dtype=torch.float)
    x = torch.unsqueeze(x, dim=0)
    x = torch.unsqueeze(x, dim=0)
    x = x.to(device)
    return x

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        #spk_out, mem_out = net(data)
        spk_out= net(data)
        spk_rec.append(spk_out)
        #mem_rec.append(mem_out)

    #return torch.stack(spk_rec), torch.stack(mem_rec)
    return torch.stack(spk_rec)


# 定义一个正则化系数
lambda_reg = 0.001

# 定义一个L2正则化项的函数
def l2_regularization(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2
    return l2_reg

def create_batches(data, labels, batch_size):
    """
    根据batch_size将数据和标签分成多个批次，并将每个批次的数据和标签转换为torch.Tensor。

    参数:
    data (list of numpy arrays or other iterable): 图片数据列表，每个元素是一张图片的数据。
    labels (list of ints or other iterable): 与图片数据对应的标签列表。
    batch_size (int): 每个批次的大小。

    返回:
    list of tuples: 每个元素是一个元组，包含一批次的torch.Tensor数据和标签。
    """
    assert len(data) == len(labels), "数据和标签的长度必须相同"
    num_samples = len(data)
    indices = list(range(num_samples))
    batches = []

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = [data[j] for j in batch_indices]
        # 假设data中的每个元素都是一个numpy数组，我们可以将它们堆叠成一个新的numpy数组，然后转换为torch.Tensor
        if isinstance(batch_data[0], np.ndarray):
            batch_data_tensor = torch.tensor(np.stack(batch_data), dtype=torch.float32)
        else:
            # 如果data中的元素不是numpy数组，你可能需要另一种方法来转换它们为tensor
            batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32)  # 注意：这里可能不正确，取决于batch_data的实际类型

        batch_labels = [labels[j] for j in batch_indices]
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)  # 通常标签是整数，所以我们使用torch.long

        batches.append((batch_data_tensor, batch_labels_tensor))

    return batches

def apply_random_inf(matrix, prob):
    # Create a mask of the same shape as the matrix
    mask = np.random.rand(*matrix.shape) < prob
    # Apply infinity where the mask is True, keeping original values where mask is False
    matrix[mask] = np.inf
    return matrix


def generate_augmented_batch(batch, n, prob):
    N, H, W = batch.shape
    # Initialize the output array with the same dtype as the input batch
    augmented_batch = np.zeros((N, n, H, W))

    for i in range(N):
        for j in range(n):
            # Copy the original matrix for each n iteration
            augmented_matrix = np.copy(batch[i])
            # Apply the random inf transformation
            augmented_matrix = apply_random_inf(augmented_matrix, prob)
            # Assign the transformed matrix to the corresponding position in the output array
            augmented_batch[i, j] = augmented_matrix

    mask = np.isinf(augmented_batch)
    augmented_batch[mask] = 0.0

    return augmented_batch


def compute_contrastive_loss(sim_matrix, labels, temperature=0.5, device=None):
    """
    Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss given a similarity matrix and labels.

    Parameters:
    sim_matrix (torch.Tensor): The cosine similarity matrix with shape [batch_size, batch_size].
    labels (torch.Tensor): The labels with shape [batch_size].
    temperature (float): The temperature scaling factor. Default is 0.5.
    device (str or torch.device, optional): The device to use for tensor operations. Default is None, which means using the same device as the input tensors.

    Returns:
    torch.Tensor: The NT-Xent loss.
    """

    # Ensure tensors are on the same device
    if device is None:
        device = sim_matrix.device

    sim_matrix = sim_matrix.to(device)
    labels = labels.to(device)

    batch_size = sim_matrix.size(0)

    # Create mask matrix
    mask = torch.eye(batch_size, dtype=torch.bool).to(device)

    # Compute positive and negative mask
    #pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool() & ~mask
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
    neg_mask = ~pos_mask & ~mask


    # Extract positive and negative similarities
    pos_sim = sim_matrix[pos_mask]
    neg_sim = sim_matrix[neg_mask]
    # print(pos_sim)
    # print(neg_sim)

    # Compute logits
    logits_pos = pos_sim / temperature
    logits_neg = neg_sim / temperature
    # Compute positive loss
    #pos_loss = -torch.log(torch.softmax(logits_pos, dim=1)[torch.arange(batch_size), labels_pos])
    pos_loss = -torch.logsumexp(logits_pos,dim = -1)
    # Compute negative loss using log-sum-exp trick
    neg_loss = torch.logsumexp(logits_neg,dim = -1)
    # Total loss
    contrastive = torch.sigmoid(pos_loss + neg_loss)  # Dividing by 2 is optional, depending on your preference for scaling

    return contrastive


# 固定随机数种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # NumPy的种子

# 示例：设置种子
set_seed(40)
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

# net = nn.Sequential(nn.Conv2d(10, 16, 3),
#                     #nn.MaxPool2d(2),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     nn.Conv2d(16, 32, 3),
#                     #nn.MaxPool2d(2),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     nn.Flatten(),
#                     nn.Linear(1016000, 1024),
#                     nn.Linear(1024,64)
#                     #snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
#                     ).to(device)

class scnn(nn.Module):
    def __init__(self):
        super(scnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(381000, 64)

        self._initialize_weights()

    def forward(self, input):
        x = self.conv1(input)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
#加载预训练模型
#net.load_state_dict(torch.load("scnn_100e.pt"))
net = scnn().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
train_num = 1
num_epochs = 200
batch = 30#batch大小需要能够整除120
total_acc = 0
train_data = sio.loadmat('train_matrix.mat')["matrix"]
test_data = sio.loadmat('test_matrix.mat')["matrix"]
all_data = sio.loadmat("alldataNew.mat")
train_labels = all_data["trainLables"][0]
test_labels = all_data["testLables"][0]

batches = create_batches(train_data, train_labels, batch)
loss_fn = SF.ce_rate_loss()

net._initialize_weights()
for epo in tqdm(range(num_epochs)):
    net.train()
    for batch_data, batch_labels in batches:
        aug_data1 = generate_augmented_batch(batch_data,3,0.1)
        aug_data_tensor1 = torch.from_numpy(aug_data1).to(torch.float32)
        aug_data2 = generate_augmented_batch(batch_data,3,0.1)
        aug_data_tensor2 = torch.from_numpy(aug_data2).to(torch.float32)
        utils.reset(net)
        output1 = net(aug_data_tensor1.to("cuda"))
        output2 = net(aug_data_tensor2.to("cuda"))

        #sim = F.cosine_similarity(output1.detach(), output2, dim=-1)
        sim = F.cosine_similarity(output1.unsqueeze(1), output2.unsqueeze(0), dim=-1)
        loss = compute_contrastive_loss(sim,batch_labels,0.5,"cuda")
        #reg_loss为正则化项
        reg_loss = lambda_reg * l2_regularization(net)
        total_loss = loss + reg_loss
        #total_loss = loss
        # print(total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

#summary(net, input_size=(3, 129, 258), device="cuda")
print(net.state_dict().keys())

#剪枝代码
prune.l1_unstructured(net.conv1, name="weight", amount=0.3)
prune.l1_unstructured(net.conv2, name="weight", amount=0.1)
prune.l1_unstructured(net.fc1, name="weight", amount=0.1)

net.eval()
train_batch = create_batches(train_data, train_labels, 120)
test_batch = create_batches(test_data, test_labels, 40)
train_batch_data, train_batch_labels = train_batch[0]
test_batch_data, test_batch_labels = test_batch[0]

train_aug_data = generate_augmented_batch(train_batch_data, 3, 0)
test_aug_data = generate_augmented_batch(test_batch_data, 3, 0)

train_aug_data_tensor = torch.from_numpy(train_aug_data).to(torch.float32)
test_aug_data_tensor = torch.from_numpy(test_aug_data).to(torch.float32)

utils.reset(net)
train_output = net(train_aug_data_tensor.to(device))
utils.reset(net)
test_output = net(test_aug_data_tensor.to(device))

# # 训练SVM模型
train_output = train_output.cpu()
test_output = test_output.cpu()
np_train_output = train_output.detach().numpy()
np_test_output = test_output.detach().numpy()
train_batch_labels = train_batch_labels.cpu()
np_train_batch_labels = train_batch_labels.detach().numpy()
test_batch_labels = test_batch_labels.cpu()
np_test_batch_labels = test_batch_labels.detach().numpy()
# 合并训练集和测试集以便一起应用t-SNE
X_combined = np.vstack((np_train_output, np_test_output))
y_combined = np.hstack((np_train_batch_labels, np_test_batch_labels))

clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(np_train_output, np_train_batch_labels)

labels = [
    "1", "2", "3", "4",
    "5", "6", "7", "8"
]

# 对整个数据集应用t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=91)
X_tsne_combined = tsne.fit_transform(X_combined)

# 分离出训练集和测试集的降维结果
X_tsne_train = X_tsne_combined[:len(np_train_output)]
X_tsne_test = X_tsne_combined[len(np_train_output):]
# 预测和评估模型
y_pred = clf.predict(np_test_output)


# 4. 计算混淆矩阵
conf_mat = confusion_matrix(np_test_batch_labels, y_pred)
# 5. 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))  # 设置图的大小
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
disp.plot(cmap='Blues', ax=ax, colorbar=True)

# 设置标题和显示
plt.title("Confusion Matrix for SVM Classifier")
plt.tight_layout()
plt.show()

# 打印分类报告和准确率
print("1111111111111111111111111111111111111111111111111111111111111111111")
print(classification_report(np_test_batch_labels, y_pred))
print("Accuracy:", accuracy_score(np_test_batch_labels, y_pred))
total_acc += accuracy_score(np_test_batch_labels, y_pred)
