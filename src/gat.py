import torch
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
    def __init__(self, num_features, num_classes, num_heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # First GAT layer
        self.gat1 = GATConv(num_features, 8, heads=num_heads, dropout=dropout)
        
        # Second GAT layer
        self.gat2 = GATConv(8 * num_heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Apply dropout to the input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

# Initialize the model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# Move data to the device
data = data.to(device)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def test():
    model.eval()
    out = model(data.x, data.edge_index)
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
plt.savefig("figs/cora_graph_structure.png")  # Save the plot to a figure
plt.show()

# Visualize class distribution
class_counts = data.y.unique(return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(class_counts[0].tolist(), class_counts[1].tolist(), tick_label=class_counts[0].tolist())
plt.xlabel('Class')
plt.ylabel('Number of Nodes')
plt.title('Class Distribution in Cora Dataset')
plt.savefig("figs/cora_class_distribution.png")  # Save the plot to a figure
plt.show()