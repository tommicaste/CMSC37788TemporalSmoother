from open import get_HCP_Age, get_HCP_Gender
from neuro import DynamicGNN
from tempsmooth import DynamicSpatialTemporalClassifier

from train import train_model

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


# Set path

path= ''

# Optimized models hyperparameters 

#Temp smooth gender
tempsmooth_gender = DynamicSpatialTemporalClassifier(
    in_channels=100,         
    embedding_sizes=[64],   
    mlp_sizes=[32],      
    num_classes=2,           
    conv_operator=GCNConv,   
    attn_heads=[1],          
    activation=nn.ReLU,      
    pooling='mean'           
)

#Temp smooth age
tempsmooth_age = DynamicSpatialTemporalClassifier(
    in_channels=100,         
    embedding_sizes=[64],   
    mlp_sizes=[32, 16],      
    num_classes=3,           
    conv_operator=GCNConv,   
    attn_heads=[2],          
    activation=nn.ReLU,      
    pooling='mean'           
)

# Neuro Gender
neuro_gender = DynamicGNN(
    input_dim=100,
    hidden_channels=64,
    hidden_dim=128,
    num_heads=8,
    num_layers=3,
    GNN=GCNConv,
    dropout=0.2,
    num_classes=2
)

# Neuro Age

neuro_age = DynamicGNN(
    input_dim=100,
    hidden_channels=64,
    hidden_dim=64,
    num_heads=4,
    num_layers=3,
    GNN=GCNConv,
    dropout=0.2,
    num_classes=2
)

# get data
train_df_gender, test_df_gender = get_HCP_Gender(path)
train_df_age, test_df_age = get_HCP_Age(path)


#Train the model and get testing validation score at each iteration
trained_tempsmooth_gender = train_model(tempsmooth_gender, train_df_gender, test_df_gender)

trained_tempsmooth_age = train_model(tempsmooth_age, train_df_age, test_df_age)

trained_neuro_gender = train_model(neuro_gender, train_df_gender, test_df_gender)

trained_neuro_age = train_model(neuro_age, train_df_age, test_df_age)

