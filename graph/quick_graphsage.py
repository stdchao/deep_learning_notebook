import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import Planetoid

from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
# from model.gcn import GCN_MPNN as GCNConv

# define network
class quickGCN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_size=16):
        super(quickGCN, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_size)
        self.conv2 = SAGEConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        output = F.log_softmax(x, dim=1)

        return output

# global var
torch.manual_seed(1234)
device = torch.device('cuda:6')

# load dataset Cora
dataset = Planetoid(root='/nfs_baoding/target_discovery/project/github/graph_learning/data/pyg', name='Cora')
data = dataset[0].to(device)

# neighbor sampler
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[5, 2], 
                               batch_size=1, shuffle=True, num_workers=4)

# build model and optimizer
model = quickGCN(num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)
optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

# train
def train():
    model.train()
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_pred = out.argmax(axis=-1)
        correct = y_pred.eq(data.y)
        train_acc = correct[data.train_mask].sum().float() / data.train_mask.sum()
        val_acc = correct[data.val_mask].sum().float() / data.val_mask.sum()
        test_acc = correct[data.test_mask].sum().float() / data.test_mask.sum()
    
    return train_acc, val_acc, test_acc

# train loop
for epoch in range(200):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print('Epoch {:2d}, Loss {:.4f}, Train acc {:.2f}%, Val acc {:.2f}%, Test acc {:.2f}%'.format(
        epoch, loss, 100*train_acc, 100*val_acc, 100*test_acc
    ))