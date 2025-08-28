import torch
import numpy as np
from time import time
from sklearn.datasets import load_svmlight_files 
import math
from nn_utils import *

import sys

seed = int(sys.argv[4])  # 86774275 19557715 16036938 13759845 10696796
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

train_path = "train_data_path/data_name/data_name.train" # path removed to preserve anonymity
val_path = "heldout_data_path/data_name/data_name.heldout" # path removed to preserve anonymity
test_path = "test_data_path/data_name/data_name.test" # path removed to preserve anonymity
X_train, y_train, X_val, y_val, X_test, y_test = load_svmlight_files((train_path, val_path, test_path), dtype=np.float32, multilabel=False)

print("Experiments on Dmoz")
print("num of training data: ", X_train.shape[0])
print("num of validation data: ", X_val.shape[0])
print("num of test data: ", X_test.shape[0])
print("num of features: ", X_train.shape[1])

num_classes = len(set(y_train.tolist() + y_val.tolist() + y_test.tolist()))
y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)

num_features = X_train.shape[1]

class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, embed_dim, last_layer_complex=False):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
        )
        self.last_layer_complex = last_layer_complex
        if last_layer_complex:
            self.last_layer = torch.nn.Linear(hidden_dim, embed_dim, dtype=torch.complex128)
        else:
            self.last_layer = torch.nn.Linear(hidden_dim, embed_dim)
    def forward(self, x):
        x = self.model(x)
        if self.last_layer_complex:
            x = x.type(torch.complex128)
        x = self.last_layer(x)
        row_norms = torch.norm(x, dim=1, p=2).unsqueeze(1)
        return x/row_norms

train_loader = torch.utils.data.DataLoader(sparse_dataset(X_train, y_train), batch_size=256, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)
test_loader = torch.utils.data.DataLoader(sparse_dataset(X_test, y_test), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)
val_loader = torch.utils.data.DataLoader(sparse_dataset(X_val, y_val), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sparse_collate_coo)

device = torch.device("cuda")

def run_exp(embed_dim, embed_type, epochs):
    label_embed = generate_label_embedding(embed_type, num_classes, embed_dim, rng, device)

    def loss_f(output, target):
        return ((output-target).conj() * (output-target)).real.sum()
    model = Net(num_features=num_features, hidden_dim=2500, embed_dim=embed_dim, last_layer_complex=('complex' in embed_type)).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=0e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)

    epoch_time_hist = []
    train_time = 0
    val_loss_hist = []
    val_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(1, epochs+1):
        start = time()
        train_le(model, label_embed, loss_f, device, train_loader, optimizer, epoch)
        scheduler.step()
        train_time += time() - start
        val_loss, val_acc = test_le(model, label_embed, loss_f, device, val_loader)
        print("validation results. l2_loss: {:.6f}, accuracy: {:.4f}".format(val_loss, val_acc))
        test_loss, test_acc = test_le(model, label_embed, loss_f, device, test_loader)
        print("test results. l2_loss: {:.6f}, accuracy: {:.4f}".format(test_loss, test_acc))
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        epoch_time_hist.append(train_time)

    # measure prediction time:
    prediction_start = time()
    test_loss, test_acc = test_le(model, label_embed, loss_f, device, test_loader)
    prediction_time = time() - prediction_start
    return val_loss_hist, val_acc_hist, test_loss_hist, test_acc_hist, epoch_time_hist, prediction_time

val_loss_hist, val_acc_hist, test_loss_hist, test_acc_hist, epoch_time_hist, prediction_time = run_exp(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
print("validation loss: ", val_loss_hist)
print("validation accuracy: ", val_acc_hist)
print("test loss: ", test_loss_hist)
print("test accuracy: ", test_acc_hist)
print("training time by epoch = ", epoch_time_hist)
print("prediction time = ", prediction_time)
