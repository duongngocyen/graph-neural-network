import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.tensorboard import SummaryWriter

# Load the Cora dataset
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]

# Define the GAT model
class GAT(torch.nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, data):
        return self.gat_net(data)

# Define the GAT layer implementations
class GATLayer(torch.nn.Module):
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        if layer_type == LayerType.IMP1:
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params(layer_type)

    def init_params(self, layer_type):
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients

        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

class GATLayerImp3(GATLayer):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        neighborhood_sums = torch.zeros(exp_scores_per_edge.shape, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        return exp_scores_per_edge / (neighborhood_sums.index_select(self.nodes_dim, trg_index) + 1e-16).unsqueeze(-1)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        return this.expand_as(other)

class GATLayerImp2(GATLayer):
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP2, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        in_nodes_features, connectivity_mask = data
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1, keepdim=True)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1, keepdim=True)

        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)

        all_scores = self.leakyReLU(scores_source + scores_target)
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)

        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)

class GATLayerImp1(GATLayer):
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP1, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        in_nodes_features, connectivity_mask = data
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = torch.matmul(in_nodes_features.unsqueeze(0), self.proj_param)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)

        all_scores = self.leakyReLU(scores_source + scores_target.transpose(1, 2))
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)

        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj)
        out_nodes_features = out_nodes_features.transpose(0, 1)

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)

# Helper functions
def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayerImp1
    elif layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')

# Initialize the model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(num_of_layers=2, num_heads_per_layer=[8, 1], num_features_per_layer=[dataset.num_features, 8, dataset.num_classes]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# Move data to the device
data = data.to(device)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model((data.x, data.edge_index))
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def test():
    model.eval()
    out = model((data.x, data.edge_index))
    pred = out.argmax(dim=1)
    accs = []
    f1_scores = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        f1 = f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro')
        accs.append(acc)
        f1_scores.append(f1)
    return accs, f1_scores

# Initialize TensorBoard writer
writer = SummaryWriter()

# Training loop
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()[0]
    train_f1, val_f1, test_f1 = test()[1]
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('F1 Score/train', train_f1, epoch)
    writer.add_scalar('F1 Score/val', val_f1, epoch)
    writer.add_scalar('F1 Score/test', test_f1, epoch)

# Close TensorBoard writer
writer.close()

# Exploratory Data Analysis (EDA)
print("\nExploratory Data Analysis (EDA):")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features per node: {data.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")

# Visualize the graph structure
G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())
plt.figure(figsize=(12, 8))
nx.draw(G, node_size=20, node_color='blue', edge_color='gray', with_labels=False)
plt.title("Cora Dataset Graph Structure")
plt.savefig("cora_graph_structure.png")  # Save the plot to a figure
plt.show()

# Visualize class distribution
class_counts = data.y.unique(return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(class_counts[0].tolist(), class_counts[1].tolist(), tick_label=class_counts[0].tolist())
plt.xlabel('Class')
plt.ylabel('Number of Nodes')
plt.title('Class Distribution in Cora Dataset')
plt.savefig("cora_class_distribution.png")  # Save the plot to a figure
plt.show()