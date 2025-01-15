import numpy as np
import scipy.io as sio
# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
from torchsummary import summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
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

train_data = sio.loadmat('train_matrix.mat')
test_data = sio.loadmat('test_matrix.mat')
all_data = sio.loadmat("alldataNew.mat")
targets_labels = all_data["trainLables"][0]
targets_labels = torch.from_numpy(targets_labels)
test_labels = all_data["testLables"][0]
test_labels = torch.from_numpy(test_labels)
print(test_labels)

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 10

# net = nn.Sequential(nn.Conv2d(1, 6, 3),
#                     #nn.MaxPool2d(2),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     nn.Conv2d(6, 12, 3),
#                     #nn.MaxPool2d(2),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     nn.Flatten(),
#                     nn.Linear(381000, 8)
#                     #snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
#                     ).to(device)

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # NumPy的种子

# 示例：设置种子
set_seed(40)

# already imported snntorch.functional as SF
loss_fn = SF.ce_rate_loss()

net = scnn().to(device)
summary(net, input_size=(1, 129, 258), device="cuda")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
num_epochs = 10
loss_hist = []
test_acc_hist = []
counter = 0

net._initialize_weights()
for epoch in range(num_epochs):
    train_total = 120
    train_mean_loss = 0
    train_all_loss = 0
    train_count = 0
    # Training loop
    for i in range(len(train_data["matrix"])):
        train_data["matrix"][i][np.isinf(train_data["matrix"][i])] = 0
        input = torch.from_numpy(train_data["matrix"][i])
        input = torch.tensor(input, dtype=torch.float).to(device)
        input = torch.unsqueeze(input, dim=0)
        input = torch.unsqueeze(input, dim=0)
        input = input.to(device)
        targets_val = targets_labels[i]-1
        targets = torch.unsqueeze(targets_val, dim=0).to(device)

        # forward pass
        net.train()
        #spk_rec, _ = forward_pass(net, num_steps, ddd)
        spk_rec = forward_pass(net, num_steps, input)
        flattened_tensor = torch.flatten(spk_rec.argmax(dim=2))
        unique_elements, counts = torch.unique(flattened_tensor, return_counts=True)
        max_count, max_count_idx = counts.max(0)
        most_frequent_element = unique_elements[max_count_idx]
        if most_frequent_element == targets_val:
            train_count += 1
        #print("spr_rec",spk_rec)
        #print("targets",targets)
        # initialize the loss & sum over time
        loss_val = loss_fn(spk_rec, targets)
        reg_loss = lambda_reg * l2_regularization(net)
        total_loss = loss_val + reg_loss
        train_all_loss += total_loss
        #print("loss", loss_val)
        #print("11111111111111111111111111111111111111111111111111")
        # Gradient calculation + weight update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

    train_mean_loss = train_all_loss / train_total
    print(f"train loss: {train_mean_loss}\n")
    print("train acc", train_count / train_total)

    #Test set
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
            #统计出输出结果中得票数最高的类别
            flattened_tensor = torch.flatten(spk_rec.argmax(dim=2))
            unique_elements, counts = torch.unique(flattened_tensor, return_counts=True)
            max_count, max_count_idx = counts.max(0)
            most_frequent_element = unique_elements[max_count_idx]
            # print("test_spr_rec",spk_rec.argmax(dim=2))
            # print("arg_max", most_frequent_element)
            # print("test_targets",test_targets)
            if most_frequent_element == test_targets_val:
                count += 1
            # initialize the loss & sum over time
            all_loss += loss_fn(spk_rec, test_targets)

        mean_loss = all_loss / total
        print(f"Test loss: {mean_loss}\n")
        print("test acc", count /total)
        print("1111111111111111111111111111111111111111111111111111")


