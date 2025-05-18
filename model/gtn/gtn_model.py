from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GTN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_feats=1024, num_layers=2, dropout=0.5):
        super(GTN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(num_features, hidden_feats, heads=1, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_feats, hidden_feats, heads=1, dropout=dropout))
        self.fc = nn.Linear(hidden_feats, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
                print(f'x shape after conv: {x.shape}')
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
if __name__ == "__main__":
    # Example usage
    model = GTN(num_features=100, num_classes=10)
    print(model)
    # Dummy data
    import torch
    x = torch.randn(32, 100)  # 32 samples, 100 features
    edge_index = torch.randint(0, 32, (2, 100))  # 1000 edges
    edge_attr = torch.randn(100).reshape(-1, 1)  # 100 edges with 1 feature each
    print(edge_index)  # Should be (2, 1000)
    out = model(x, edge_index, edge_attr)
    print(out.shape)  # Should be (32, 10) for 32 samples and 10 classes