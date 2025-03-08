from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv,SAGEConv,GraphConv,TransformerConv,ChebConv,GATConv,SGConv,GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import math
from tqdm import tqdm
import torch.optim as optim


class SpatialConvLayer(nn.Module):
    """
    Spatial custom convolution layer to take multiple convolution options
    """
    def __init__(self, in_channels, out_channels, conv_operator, activation=nn.ReLU()):
        """
        Args:
            in_channels : Dimension of the input node features
            out_channels : Dimension of the output node features
            conv_operator : GNN operator
            activation : Defaults to ReLU()
        """
        super(SpatialConvLayer, self).__init__()
        self.conv = conv_operator(in_channels, out_channels)
        self.activation = activation

    def forward(self, x, edge_index):
        # Apply the chosen graph convolution operator
        x = self.conv(x, edge_index)
        # Apply the activation function
        if self.activation is not None:
            x = self.activation(x)
        return x
    

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer Function to merge embeddings
    """
    def __init__(self, embed_dim, num_heads=1, num_layers=1):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, H_tilde, H_prev):
        """
        Args:
            H_tilde : New embeddings at time t, shape [N, embed_dim]
            H_prev : Embeddings from time t-1, shape [N, embed_dim]
        Returns:
            H_out : Fused embeddings, shape [N, embed_dim]
        """
        # Stack H_prev and H_tilde along the sequence dimension.
        x = torch.stack([H_prev, H_tilde], dim=1)
        
        # Apply the transformer
        out = self.transformer(x)
        
        H_out = out.mean(dim=1)
        return H_out



class DynamicSpatialTemporalClassifier(nn.Module):
    """
    A joint model that:
      1) Processes a sequence of snapshots via multiple (Conv -> Transformer) layers
         to produce final node embeddings for the last snapshot
      2) Pools the final node embeddings and passes the pooled vector through an MLP
         to output class logits
    """
    def __init__(
        self,
        in_channels,        
        embedding_sizes,     
        mlp_sizes,           
        num_classes,         
        conv_operator=GCNConv,
        attn_heads=None,     
        activation=nn.ReLU,
        pooling='mean'       
    ):
        super(DynamicSpatialTemporalClassifier, self).__init__()
        
        # attention head per layer
        if attn_heads is None:
            attn_heads = [1] * len(embedding_sizes)
        if len(attn_heads) < len(embedding_sizes):
            attn_heads += [1] * (len(embedding_sizes) - len(attn_heads))
        
        self.num_layers = len(embedding_sizes)
        self.layers_conv = nn.ModuleList()
        self.layers_attn = nn.ModuleList()
        current_in = in_channels
        
        # build the convolution layers 
        for i, out_dim in enumerate(embedding_sizes):
            self.layers_conv.append(
                SpatialConvLayer(
                    in_channels=current_in,
                    out_channels=out_dim,
                    conv_operator=conv_operator,
                    activation=activation()
                )
            )
            self.layers_attn.append(
                TemporalTransformer(embed_dim=out_dim, num_heads=attn_heads[i])
            )
            current_in = out_dim
        
        # Classification head
        self.pooling = pooling
        mlp_layers = []
        prev_dim = embedding_sizes[-1]  # Final embedding dimension.
        for hidden_dim in mlp_sizes:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(activation())
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, snapshots):
        """
        Args:
            snapshots: List of PyGeometric Data objects 
        Returns:
            logits : Classification logits with shape [1, num_classes]
        """
        # Initialize previous embeddings for each layer.
        prev_embeddings = [None] * self.num_layers
        
        # Process snapshots in chronological order.
        for t, data in enumerate(snapshots):
            x = data.x
            edge_index = data.edge_index
            # Process through each layer.
            for i in range(self.num_layers):
                
                tilde = self.layers_conv[i](x, edge_index)
                
                if t == 0: # for t = 0
                    h = tilde
                else:# for t>0 
                    h = self.layers_attn[i](tilde, prev_embeddings[i])
                x = h
                prev_embeddings[i] = h
        
        if self.pooling == 'mean':
            pooled = x.mean(dim=0) 
        elif self.pooling == 'max':
            pooled, _ = x.max(dim=0) 
        pooled = pooled.unsqueeze(0)  # [1, embed_dim]
        h_mlp = self.mlp(pooled)       # [1, mlp_last]
        logits = self.output_layer(h_mlp)  # [1, num_classes]
        return logits
