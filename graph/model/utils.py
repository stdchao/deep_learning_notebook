import torch

def add_self_loops(edge_index, num_nodes):
    '''
    implement torch_geometric.utils.add_self_loops
    '''
    self_loops_index = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    self_loops_index = self_loops_index.repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops_index], dim=1)
    return edge_index

def degree(index, num_nodes):
    '''
    implement torch_geometric.utils.degree
    '''
    deg = torch.zeros(num_nodes, dtype=index.dtype, device=index.device)
    deg = deg.scatter_add_(0, index, torch.ones(index.size(0), dtype=index.dtype, device=index.device))
    return deg