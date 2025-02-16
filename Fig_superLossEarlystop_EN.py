import numpy as np
import scipy.io as sio
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Save_Figure import *



rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16
# 定义正则化系数
lambda_reg = 0.001

# 定义L2正则化项的函数
def l2_regularization(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2
    return l2_reg

# 加载数据
train_data = sio.loadmat('train_matrix.mat')
test_data = sio.loadmat('test_matrix.mat')
all_data = sio.loadmat("alldataNew.mat")
targets_labels = all_data["trainLables"][0]
targets_labels = torch.from_numpy(targets_labels)
test_labels = all_data["testLables"][0]
test_labels = torch.from_numpy(test_labels)

# 设置设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 神经元和仿真参数
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 10

import itertools
def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        #spk_out, mem_out = net(data)
        spk_out= net(data)
        spk_rec.append(spk_out)
        #mem_rec.append(mem_out)

    return torch.stack(spk_rec)

# 定义模型
class scnn(nn.Module):
    def __init__(self):
        super(scnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(254000, 8)

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

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# 设置随机种子
set_seed(40)

# 损失函数
loss_fn = SF.ce_rate_loss()

# 初始化模型
net = scnn().to(device)

# 显示模型摘要
from torchsummary import summary
summary(net, input_size=(1, 129, 258), device="cuda")

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

# 训练和测试过程中的损失和准确率记录
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Early stopping settings
patience = 5  # How many epochs to wait for improvement
best_loss = float('inf')
epochs_without_improvement = 0
early_stop_epoch = -1

# 训练过程
num_epochs = 100
for epoch in range(num_epochs):
    train_total = 120
    train_mean_loss = 0
    train_all_loss = 0
    train_count = 0

    # 训练循环
    for i in range(len(train_data["matrix"])):
        train_data["matrix"][i][np.isinf(train_data["matrix"][i])] = 0
        input = torch.from_numpy(train_data["matrix"][i])
        input = torch.tensor(input, dtype=torch.float).to(device)
        input = torch.unsqueeze(input, dim=0)
        input = torch.unsqueeze(input, dim=0)
        input = input.to(device)
        targets_val = targets_labels[i] - 1
        targets = torch.unsqueeze(targets_val, dim=0).to(device)

        # 前向传播
        net.train()
        spk_rec = forward_pass(net, num_steps, input)

        # 计算训练准确率
        flattened_tensor = torch.flatten(spk_rec.argmax(dim=2))
        unique_elements, counts = torch.unique(flattened_tensor, return_counts=True)
        max_count, max_count_idx = counts.max(0)
        most_frequent_element = unique_elements[max_count_idx]
        if most_frequent_element == targets_val:
            train_count += 1

        # 计算损失
        loss_val = loss_fn(spk_rec, targets)
        reg_loss = lambda_reg * l2_regularization(net)
        total_loss = loss_val + reg_loss
        train_all_loss += total_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_mean_loss = train_all_loss / train_total
    train_accuracies.append(train_count / train_total)
    train_losses.append(train_mean_loss)

    print(f"train loss: {train_mean_loss}\n")
    print("train acc", train_count / train_total)

    # 测试过程
    with torch.no_grad():
        total = 40
        mean_loss = 0
        all_loss = 0
        net.eval()
        count = 0
        for i in range(len(test_data["matrix"])):
            test_data["matrix"][i][np.isinf(test_data["matrix"][i])] = 0
            test_input = torch.from_numpy(test_data["matrix"][i])
            test_input = torch.tensor(test_input, dtype=torch.float).to(device)
            test_input = torch.unsqueeze(test_input, dim=0)
            test_input = torch.unsqueeze(test_input, dim=0)
            test_targets_val = test_labels[i] - 1
            test_targets = torch.unsqueeze(test_targets_val.to(device), dim=0)

            spk_rec = forward_pass(net, num_steps, test_input)

            # 计算测试准确率
            flattened_tensor = torch.flatten(spk_rec.argmax(dim=2))
            unique_elements, counts = torch.unique(flattened_tensor, return_counts=True)
            max_count, max_count_idx = counts.max(0)
            most_frequent_element = unique_elements[max_count_idx]
            if most_frequent_element == test_targets_val:
                count += 1

            # 计算损失
            all_loss += loss_fn(spk_rec, test_targets)

        mean_loss = all_loss / total
        test_losses.append(mean_loss)
        test_accuracies.append(count / total)

        print(f"Test loss: {mean_loss}\n")
        print("test acc", count / total)
        print("-------------------------------------------------------")

        # Check early stopping condition
        if mean_loss < best_loss:
            best_loss = mean_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            early_stop_epoch = epoch
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Early stopping break
    if early_stop_epoch != -1:
        break

# 绘制损失和准确率
epochs = range(len(train_losses))
# Convert the tensors to CPU and then to numpy arrays before plotting
train_losses_cpu = torch.tensor(train_losses).cpu().numpy()
test_losses_cpu = torch.tensor(test_losses).cpu().numpy()

train_accuracies_cpu = torch.tensor(train_accuracies).cpu().numpy()
test_accuracies_cpu = torch.tensor(test_accuracies).cpu().numpy()

experiment_shadow = train_accuracies_cpu + 0.02
model_shadow = test_accuracies_cpu + 0.02



# 绘制训练损失和准确率
# 创建 1 行 2 列的子图，返回 fig 和 axes 对象
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 绘制损失图
axes[0].plot(epochs, train_losses_cpu, label='Training Loss', color='blue')
axes[0].plot(epochs, test_losses_cpu, label='Test Loss', color='red')
axes[0].set_xlabel('Epochs')  # X轴标签为英文
axes[0].set_ylabel('Loss')  # Y轴标签为英文
axes[0].legend()

# 绘制准确率图
axes[1].plot(epochs, train_accuracies_cpu, label='Training Accuracy', color='blue')
axes[1].plot(epochs, test_accuracies_cpu, label='Test Accuracy', color='red')

axes[1].set_xlabel('Epochs')  # X轴标签为英文
axes[1].set_ylabel('Accuracy')  # Y轴标签为英文
axes[1].legend()

# 绘制阴影区域
plt.fill_between(epochs, train_accuracies_cpu, model_shadow, color='lightblue', alpha=0.5)
plt.fill_between(epochs, test_accuracies_cpu, experiment_shadow, color='lightblue', alpha=0.3)

# 标出早停点
if early_stop_epoch != -1:
    axes[0].axvline(x=early_stop_epoch, color='green', linestyle='--', label='Early Stopping')

plt.tight_layout()
plt.show()

# 保存为 PDF（可选）
current_file = os.path.basename(__file__)
save_dir = "Python_Figures_EN"
os.makedirs(save_dir, exist_ok=True)
filename = os.path.splitext(current_file)[0] + ".pdf"
save_figure_as_pdf(fig, save_dir, filename)

print(f"Figure saved to: {filename}")
