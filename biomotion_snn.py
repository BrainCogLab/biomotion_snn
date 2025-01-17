import scipy.io as sio
import os
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from sklearn.metrics import  classification_report
from sklearn import svm
import torch.nn.utils.prune as prune
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

def forward_pass(net, num_steps, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out= net(data)
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)

def l2_regularization(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2
    return l2_reg

def create_batches(data, labels, batch_size):
    num_samples = len(data)
    indices = list(range(num_samples))
    batches = []

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = [data[j] for j in batch_indices]
        if isinstance(batch_data[0], np.ndarray):
            batch_data_tensor = torch.tensor(np.stack(batch_data), dtype=torch.float32)
        else:
            batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32)

        batch_labels = [labels[j] for j in batch_indices]
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

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
    if device is None:
        device = sim_matrix.device

    sim_matrix = sim_matrix.to(device)
    labels = labels.to(device)

    batch_size = sim_matrix.size(0)

    # Create mask matrix
    mask = torch.eye(batch_size, dtype=torch.bool).to(device)

    # Compute positive and negative mask
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
    neg_mask = ~pos_mask & ~mask


    # Extract positive and negative similarities
    pos_sim = sim_matrix[pos_mask]
    neg_sim = sim_matrix[neg_mask]

    # Compute logits
    logits_pos = pos_sim / temperature
    logits_neg = neg_sim / temperature
    # Compute positive loss
    pos_loss = -torch.logsumexp(logits_pos,dim = -1)
    # Compute negative loss
    neg_loss = torch.logsumexp(logits_neg,dim = -1)
    # Total loss
    contrastive = torch.sigmoid(pos_loss + neg_loss)  # Dividing by 2 is optional, depending on your preference for scaling

    return contrastive

def save_figure_as_pdf(figure, save_dir, filename):
    # Remove the file extension
    file_name_without_extension = os.path.splitext(filename)[0]

    # Get the current path and parent path
    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)

    # Define the save path
    save_path = os.path.join(parent_path, save_dir)

    # If the save path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print("The figure already exists")

    # Save the figure
    save_file_path = os.path.join(save_path, filename)
    figure.savefig(save_file_path, dpi=300, bbox_inches='tight', format='pdf')

    print("Diagram saved successfully at:", save_file_path)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

t0 = time.time()

set_seed(40)
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

lambda_reg = 0.001
train_num = 1
num_epochs = 200
batch = 30
total_acc = 0
n = 3
# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

class scnn(nn.Module):
    def __init__(self):
        super(scnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
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

net = scnn().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

train_data = sio.loadmat('train_matrix.mat')["matrix"]
test_data = sio.loadmat('test_matrix.mat')["matrix"]
all_data = sio.loadmat("alldataNew.mat")
train_labels = all_data["trainLables"][0]
test_labels = all_data["testLables"][0]

batches = create_batches(train_data, train_labels, batch)

net._initialize_weights()
for epo in tqdm(range(num_epochs)):
    net.train()
    for batch_data, batch_labels in batches:
        aug_data1 = generate_augmented_batch(batch_data,n,0.1)
        aug_data_tensor1 = torch.from_numpy(aug_data1).to(torch.float32)
        aug_data2 = generate_augmented_batch(batch_data,n,0.1)
        aug_data_tensor2 = torch.from_numpy(aug_data2).to(torch.float32)
        utils.reset(net)
        output1 = net(aug_data_tensor1.to(device))
        output2 = net(aug_data_tensor2.to(device))


        sim = F.cosine_similarity(output1.unsqueeze(1), output2.unsqueeze(0), dim=-1)
        loss = compute_contrastive_loss(sim,batch_labels,0.5,device)

        reg_loss = lambda_reg * l2_regularization(net)
        total_loss = loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

#pruning
prune.l1_unstructured(net.conv1, name="weight", amount=0.3)
prune.l1_unstructured(net.conv2, name="weight", amount=0.1)
prune.l1_unstructured(net.fc1, name="weight", amount=0.1)

net.eval()
train_batch = create_batches(train_data, train_labels, 120)
test_batch = create_batches(test_data, test_labels, 40)
train_batch_data, train_batch_labels = train_batch[0]
test_batch_data, test_batch_labels = test_batch[0]

train_aug_data = generate_augmented_batch(train_batch_data, n, 0)
test_aug_data = generate_augmented_batch(test_batch_data, n, 0)

train_aug_data_tensor = torch.from_numpy(train_aug_data).to(torch.float32)
test_aug_data_tensor = torch.from_numpy(test_aug_data).to(torch.float32)

utils.reset(net)
train_output = net(train_aug_data_tensor.to(device))
utils.reset(net)
test_output = net(test_aug_data_tensor.to(device))

# SVM model
train_output = train_output.cpu()
test_output = test_output.cpu()
np_train_output = train_output.detach().numpy()
np_test_output = test_output.detach().numpy()
train_batch_labels = train_batch_labels.cpu()
np_train_batch_labels = train_batch_labels.detach().numpy()
test_batch_labels = test_batch_labels.cpu()
np_test_batch_labels = test_batch_labels.detach().numpy()

labels = [
    "1", "2", "3", "4",
    "5", "6", "7", "8"
]

clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(np_train_output, np_train_batch_labels)
y_pred = clf.predict(np_test_output)

conf_mat = confusion_matrix(np_test_batch_labels, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
disp.plot(cmap='Blues', ax=ax, colorbar=True)

current_file = os.path.basename(__file__)
save_dir = "Python_Figures"
filename = os.path.splitext(current_file)[0] + ".pdf"
save_figure_as_pdf(fig, save_dir, filename)

plt.title("Confusion Matrix for SVM Classifier")
plt.tight_layout()
plt.show()

print("1111111111111111111111111111111111111111111111111111111111111111111")
print(classification_report(np_test_batch_labels, y_pred))

t1 = time.time()
training_time = t1 - t0
print(f"Training time: {training_time:.2f} seconds")
