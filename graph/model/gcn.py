import torch
import torch.nn as nn
from torch_scatter import scatter
from .utils import add_self_loops, degree

class GCN_MPNN(nn.Module):
    '''
    implement torch_geometric.nn.GCNConv
    '''
    def __init__(self, in_channels, out_channels, bias=True, agg='sum'):
        super(GCN_MPNN, self).__init__()
        self.agg = agg # sum/mean/max

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(0)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index):
        '''
        MPNN: message -> aggregate -> update
        '''
        return self.propagate(x, edge_index)

    def propagate(self, x, edge_index):
        edge_index = add_self_loops(edge_index, num_nodes=x.shape[0]) # add self loop edge
        out = self.message(x, edge_index)
        out = self.aggregate(out, edge_index)
        out = self.update(out)
        return out

    def message(self, x, edge_index):
        x = self.linear(x)

        row, col = edge_index # source_to_target
        deg = degree(col, x.shape[0]) # target degree
        deg_inv_sqrt = deg.pow(-0.5) # target norm
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] # source_to_target norm

        x_j = x[row] # x_source
        x_j = norm.view(-1, 1) * x_j # x_source norm
        return x_j

    def aggregate(self, x_j, edge_index):
        _, col = edge_index
        agg_out = scatter(x_j, col, dim=0, reduce=self.agg) # reduce source to target
        return agg_out

    def update(self, agg_out):
        if self.bias is not None:
            return agg_out + self.bias
        else:
            return agg_out
