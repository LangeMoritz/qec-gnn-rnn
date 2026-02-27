import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, GraphConv


class GNN_7(torch.nn.Module):
    """
    GNN with several consecutive GraphConv layers, whose final output is converted
    to a single graph embedding (feature vector) with global_mean_pool.
    This graph embedding is cloned 12 times, and passed to 12
    dense network which perform binary classification.
    The binary classifications represent the 12 logical equivalence classes.
    """

    def __init__(
        self,
        hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
        hidden_channels_MLP=[256, 128, 64],
        num_node_features=4,
        num_classes=12,
        manual_seed=12345,
    ):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)

        # GCN layers
        channels = [num_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Dense layers for the MLP classifier
        self.mlp_layers = nn.ModuleList()
        channels = hidden_channels_GCN[-1:] + hidden_channels_MLP
        for _ in range(num_classes):
            mlp = nn.ModuleList(
                [
                    nn.Linear(in_channels, out_channels)
                    for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
                ]
            )
            self.mlp_layers.append(mlp)

        # Output layers (one for each class)
        self.output_layers = nn.ModuleList(
            [nn.Linear(hidden_channels_MLP[-1], 1) for _ in range(num_classes)]
        )

    def forward(self, x, edge_index, batch, edge_attr):
        #  node embeddings
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(x, inplace=True)

        # graph embedding
        x = global_mean_pool(x, batch)

        # Clone and pass through MLPs
        outputs = []
        for i in range(len(self.mlp_layers)):
            x_clone = x.clone()
            for layer in self.mlp_layers[i]:
                x_clone = layer(x_clone)
                x_clone = torch.nn.functional.relu(x_clone, inplace=True)
            output = self.output_layers[i](x_clone)
            outputs.append(output)

        # Concatenate outputs along the second dimension
        outputs = torch.cat(outputs, dim=1)

        return outputs

    
class GNN_7_1head(torch.nn.Module):
    """
    GNN with several consecutive GraphConv layers, whose final output is 
    converted to a single graph embedding (feature vector) with global_mean_pool.
    This graph embedding is passed to a dense network which perform binary
    classification. The binary classifications represent the 12 logical 
    equivalence classes.
    """

    def __init__(
        self,
        hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
        hidden_channels_MLP=[256, 128, 64],
        num_node_features=2,
        num_classes=12,
        manual_seed=12345,
    ):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)

        # GCN layers
        channels = [num_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [GraphConv(in_channels, out_channels)
            for (in_channels, out_channels) in zip(channels[:-1], channels[1:])]
        )

        # Dense layers for the MLP classifier
        channels = hidden_channels_GCN[-1:] + hidden_channels_MLP
        self.mlp_layers = nn.ModuleList(
            [nn.Linear(in_channels, out_channels)
            for (in_channels, out_channels) in zip(channels[:-1], channels[1:])]
            )

        # Output layers (one for each class)
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], num_classes)

    def forward(self, x, edge_index, batch, edge_attr):
        #  node embeddings
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(x, inplace=True)

        # graph embedding
        x = global_mean_pool(x, batch)

        # pass through MLPs
        for layer in self.mlp_layers:
            x = layer(x)
            x = torch.nn.functional.relu(x, inplace=True)
        
        # output
        output = self.output_layer(x)

        return output