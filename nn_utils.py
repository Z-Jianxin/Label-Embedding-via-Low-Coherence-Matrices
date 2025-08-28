import torch
from sklearn.metrics import pairwise_distances
import numpy as np
import math
from itertools import product


class sparse_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_features = x.shape[1]
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        return self.x.indices[self.x.indptr[i]:self.x.indptr[i+1]], self.x.data[self.x.indptr[i]:self.x.indptr[i+1]], self.y[i], self.n_features 

def sparse_collate_coo(batch):
    r = []
    c = []
    vals = []
    y = []
    n_features = batch[0][-1]
    for i, (indices, data, yi, _) in enumerate(batch):
        r.extend([i] * indices.shape[0])
        c.extend(indices)
        vals.extend(data)
        y.append(yi)
    return ([r, c], vals, (len(batch), n_features)), y

class SquaredLoss(torch.nn.Module):
    def __init__(self):
        super(SquaredLoss, self).__init__()

    def forward(self, outputs, targets):
        one_hot_approx = torch.zeros_like(outputs)
        one_hot_approx.scatter_(1, targets.unsqueeze(1), 1)
        return torch.sum((outputs - one_hot_approx) ** 2)

def train_le(model, label_embed, loss_f, device, train_loader, optimizer, epoch, log_interval=50):
    model.train()
    for idx, ((locs, vals, size), y) in enumerate(train_loader):
        x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
        y_embed = torch.index_select(label_embed, 0, torch.tensor(y, dtype=torch.int32).to(device))
        optimizer.zero_grad()
        embed_out = model(x)
        loss = loss_f(embed_out, y_embed) / len(y)
        loss.backward()
        optimizer.step()
        if (idx + 1) % log_interval == 0:
            print("train epoch: {}, batch: {}/{}, loss: {:.6f}".format(epoch, idx+1, len(train_loader), loss.item()))

def find1NN_cuda(out_cuda, label_embed_cuda):
    #dist_m = torch.cdist(out_cuda.reshape(1, out_cuda.shape[0], -1), label_embed_cuda.reshape(1, label_embed_cuda.shape[0], -1))
    #dist_m = dist_m.reshape(dist_m.shape[1], -1)
    #oneNNs = torch.argmin(dist_m, dim=1)
    #gram_m = torch.matmul(out_cuda, torch.transpose(label_embed_cuda, 0, 1))
    #return torch.argmax(gram_m, dim=1)
    gram_m = torch.matmul(torch.conj(out_cuda), torch.transpose(label_embed_cuda, 0, 1)).real
    #gram_m += torch.matmul(out_cuda, torch.transpose(label_embed_cuda, 0, 1).conj())
    return torch.argmax(gram_m.real, dim=1)

def nelson_embed(p, r=2, num_classes=None):
    arr_p = [i for i in range(p)]
    if num_classes is None:
        num_classes = p ** r
    embed_dim = p
    embed_m = np.zeros((num_classes, embed_dim), dtype=np.complex128)
    power_m = np.repeat(np.arange(p).reshape((p, 1)), repeats=r, axis=1)
    for i in range(r):
        power_m[:, i] **= i+1
    idx = 0
    for a in product(arr_p, repeat=r):
        embed_m[idx, :] = (np.array(a).reshape((1, r)) * power_m).sum(axis=1)
        idx += 1
        if idx >= num_classes:
            break
    xi_p = np.exp(2j*np.pi/p)
    embed_m = (xi_p ** embed_m) / np.sqrt(embed_dim)
    print("coherence upper bound= {:.4f}".format((r-1)/np.sqrt(p)))
    return embed_m

def generate_label_embedding(embed_type, num_classes, embed_dim, rng, device):
    if embed_type == "rademacher":
        label_embed = (rng.integers(low=0, high=2, size=(num_classes, embed_dim)) * 2 - 1) / math.sqrt(embed_dim)
        label_embed = np.float32(label_embed)
    elif embed_type == "gaussian":
        label_embed = rng.normal(size=(num_classes, embed_dim)) / math.sqrt(embed_dim)
        label_embed = np.float32(label_embed)
    elif embed_type == "gaussian_complex":
        label_embed = rng.normal(size=(num_classes, embed_dim, 2))
        label_embed = np.float32(label_embed)
        label_embed = label_embed.view(np.complex128)[:, :, 0] / math.sqrt(2*embed_dim)
    elif embed_type == "nelson_complex":
        # to use nelson's construction, we need embed_dim to be a prime
        label_embed = nelson_embed(embed_dim, r=2, num_classes=num_classes)
    else:
        raise Exception("no embedding matrix of this type {:s}".format(embed_type))
    row_norms = np.linalg.norm(label_embed, axis=1)
    label_embed = label_embed / row_norms[:, np.newaxis]
    if num_classes <= 2e4:
        gram_m = label_embed.conj() @ label_embed.T-np.eye(num_classes)
    else:
        sampled = rng.choice(num_classes, size=2*10**4, replace=False)
        gram_m = label_embed[sampled,:].conj() @ label_embed[sampled,:].T-np.eye(2*10**4)
    lambd = np.max(np.abs(gram_m))
    real_lambd = np.max(np.abs(np.real(gram_m)))
    print("lambda = ", lambd, "real_lambda = ", real_lambd)
    return torch.tensor(label_embed).to(device)
            
def test_le(model, label_embed, loss_f, device, test_loader):
    model.eval()
    mean_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, ((locs, vals, size), y) in enumerate(test_loader):
            x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
            y_embed = torch.index_select(label_embed, 0, torch.tensor(y, dtype=torch.int32).to(device))
            embed_out = model(x)
            mean_loss += loss_f(embed_out, y_embed).item()
            embed_out_detached = embed_out.detach()
            preds = find1NN_cuda(embed_out_detached, label_embed).cpu().numpy()
            correct += np.sum(preds==y)
            total += preds.shape[0]
            del x, y_embed, embed_out
    return mean_loss / len(test_loader.dataset), correct/total

def train_ce(model, loss_f, device, train_loader, optimizer, epoch, log_interval=50):
    model.train()
    for idx, ((locs, vals, size), y) in enumerate(train_loader):
        x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_f(out, torch.tensor(y, dtype=torch.int64).to(device)) / len(y)
        loss.backward()
        optimizer.step()
        if (idx + 1) % 10 == 0:
            print("train epoch: {}, batch: {}/{}, loss: {:.6f}".format(epoch, idx+1, len(train_loader), loss.item()))

def test_ce(model, loss_f, device, test_loader):
    model.eval()
    mean_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, ((locs, vals, size), y) in enumerate(test_loader):
            x = torch.sparse_coo_tensor(locs, vals, size=size, dtype=torch.float32, device=device)
            out = model(x)
            mean_loss += loss_f(out, torch.tensor(y, dtype=torch.int64).to(device)).item()
            preds = out.detach().cpu().argmax(dim=1, keepdim=False).numpy()
            correct += np.sum(preds==np.array(y))
            total += preds.shape[0]
    return mean_loss / len(test_loader.dataset), correct/total