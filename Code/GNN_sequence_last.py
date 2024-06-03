'''
The code for 'Prediction of gas chromatography retention time based on multimodal learning'
by Jinglong Lin
revised at 2024.05.01
'''

import pandas as pd
from tqdm import tqdm
import warnings
import torch
import numpy as np
from compound_tools import *
warnings.filterwarnings('ignore')
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
import numpy
from mordred import Calculator, descriptors,is_missing
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from compound_tools import *
import torch.nn.functional as F
import pymysql
from torch_geometric.data import DataLoader
import pandas as pd
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import argparse
from tqdm import tqdm
# from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
import warnings
import matplotlib.pyplot as plt
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
from mordred import Calculator, descriptors, is_missing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from scipy.stats import spearmanr
import mordred
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch_sparse
from rdkit.Chem import Descriptors, MACCSkeys
import requests
import tqdm
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, GridSearchCV
from joblib import dump, load
from rdkit import Chem
import glob
from rdkit.Chem import Draw
import pandas as pd
import re
import argparse
import os
import math
import mendeleev
import numpy as np
from matplotlib import pyplot as plt
from mendeleev.fetch import fetch_table
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import taichi as ti
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
import seaborn as sns
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, PandasTools, Descriptors
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS   #交叉验证
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import matplotlib as mpl
import lightgbm as lgb
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor, Pool
import seaborn as sns
from PIL import Image
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVR, SVC
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from torch.optim import Adam
from torch.utils.data import Subset
import copy
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from scipy.stats import mode
import warnings
import shutil
import shap
import matplotlib.patches as mpatches
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from PIL import Image
from PyPDF2 import PdfMerger
from rdkit.Chem import rdmolfiles
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import numpy as np
from compound_tools import *
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
from mordred import Calculator, descriptors,is_missing
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.cluster import KMeans
import random
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from torchsummary import summary
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem, DataStructs
from gplearn.genetic import SymbolicRegressor
from pysr import PySRRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

seed = 1314
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

calc = Calculator(descriptors, ignore_3D=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = ["bond_dir", "bond_type", "is_in_ring"]
bond_float_names=["bond_length"]
bond_angle_float_names=['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)

warnings.filterwarnings("ignore", category=FutureWarning)
def q_loss(q,y_true,y_pred):
    e = (y_true-y_pred)
    return torch.mean(torch.maximum(q*e, (q-1)*e))
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding
class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
        return bond_embedding
class RBF(torch.nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = centers.reshape([1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.reshape([-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))
class BondFloatRBF(torch.nn.Module):
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names
        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (nn.Parameter(torch.arange(0, 2, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
                'prop': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
            }
        else:
            self.rbf_params = rbf_params
        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers.to(device), gamma.to(device))
            self.rbf_list.append(rbf)
            linear = torch.nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[:, i].reshape(-1, 1)
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)  #
        return out_embed
class BondAngleFloatRBF(torch.nn.Module):
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names
        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (nn.Parameter(torch.arange(0, math.pi, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
                # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_angle_float_names:
            if name == 'bond_angle':
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers.to(device), gamma.to(device))
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim)
                self.linear_list.append(linear)
            else:
                linear = nn.Linear(len(self.bond_angle_float_names) - 1, embed_dim)
                self.linear_list.append(linear)
                break

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            if name == 'bond_angle':
                x = bond_angle_float_features[:, i].reshape(-1, 1)
                rbf_x = self.rbf_list[i](x)
                out_embed += self.linear_list[i](rbf_x)
            else:
                x = bond_angle_float_features[:, 1:]
                out_embed += self.linear_list[i](x)
                break
        return out_embed
class ConditionEmbeding(torch.nn.Module):
    def __init__(self, condition_names,condition_float_names, embed_dim, rbf_params=None):
        super(ConditionEmbeding, self).__init__()
        self.condition_names = condition_names
        self.condition_float_names=condition_float_names
        if rbf_params is None:
            self.rbf_params = {
                'eluent': (nn.Parameter(torch.arange(0,1,0.1)), nn.Parameter(torch.Tensor([10.0]))),
                'grain_radian': (nn.Parameter(torch.arange(0,10,0.1)), nn.Parameter(torch.Tensor([10.0])))# (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        self.embedding_list=torch.nn.ModuleList()

        for name in self.condition_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers.to(device), gamma.to(device))
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)
        for name in self.condition_names:
            if name=='silica_surface':
                emb = torch.nn.Embedding(2 + 5, embed_dim).to(device)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                self.embedding_list.append(emb)
            elif name=='replace_basis':
                emb = torch.nn.Embedding(6 + 5, embed_dim).to(device)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                self.embedding_list.append(emb)

    def forward(self, condition):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.condition_float_names):
            x = condition[:,2*i+1]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        for i, name in enumerate(self.condition_names):
            x = self.embedding_list[i](condition[:,2*i].to(torch.int64))
            out_embed += x
        return out_embed
class GINConv(MessagePassing):

    def __init__(self, emb_dim):

        super(GINConv, self).__init__(aggr="add")  # add mean max

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                 nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))  # nn.Parameter很特殊，专门告诉优化器它可以优化

    def forward(self, x, edge_index, edge_attr):

        edge_embedding = edge_attr
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    #以下两个方法在forward中，propagate调用
    #广播
    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)  #为消息添加非线性

    def update(self, aggr_out):
        return aggr_out
class GINNodeEmbedding(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers   #GIN的层数
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder=BondEncoder(emb_dim)
        self.bond_float_encoder=BondFloatRBF(bond_float_names,emb_dim)
        self.bond_angle_encoder=BondAngleFloatRBF(bond_angle_float_names,emb_dim)
        self.convs = torch.nn.ModuleList()
        self.convs_bond_angle=torch.nn.ModuleList()
        self.convs_bond_float=torch.nn.ModuleList()
        self.convs_bond_embeding=torch.nn.ModuleList()
        self.convs_angle_float=torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_ba = torch.nn.ModuleList()
        self.convs_condition=torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.convs_bond_angle.append(GINConv(emb_dim))
            self.convs_bond_embeding.append(BondEncoder(emb_dim))
            self.convs_bond_float.append(BondFloatRBF(bond_float_names,emb_dim))
            self.convs_angle_float.append(BondAngleFloatRBF(bond_angle_float_names,emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_ba.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_atom_bond,batched_bond_angle):
        x, edge_index, edge_attr = batched_atom_bond.x, batched_atom_bond.edge_index, batched_atom_bond.edge_attr
        edge_index_ba,edge_attr_ba= batched_bond_angle.edge_index, batched_bond_angle.edge_attr
        h_list = [self.atom_encoder(x)]
        h_list_ba=[self.bond_float_encoder(edge_attr[:,3:edge_attr.shape[1]+1].to(torch.float32))+self.bond_encoder(edge_attr[:,0:3].to(torch.int64))]

        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])   #图1：节点经过GIN进行更新

            cur_h_ba=self.convs_bond_embeding[layer](edge_attr[:,0:3].to(torch.int64))+self.convs_bond_float[layer](edge_attr[:,3:edge_attr.shape[1]+1].to(torch.float32))

            cur_angle_hidden=self.convs_angle_float[layer](edge_attr_ba)

            h_ba=self.convs_bond_angle[layer](cur_h_ba, edge_index_ba, cur_angle_hidden)

            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
                h_ba = F.dropout(h_ba, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h_ba = F.dropout(F.relu(h_ba), self.drop_ratio, training=self.training)
            if self.residual:
                h += h_list[layer]
                h_ba+=h_list_ba[layer]
            h_list.append(h)
            h_list_ba.append(h_ba)
        if self.JK == "last":
            node_representation = h_list[-1]
            edge_representation = h_list_ba[-1]
        elif self.JK == "sum":
            node_representation = 0
            edge_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]
                edge_representation += h_list_ba[layer]
        return node_representation,edge_representation

class AttentionFusion(nn.Module):
    def __init__(self, emb_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
            nn.Softmax(dim=1)
        )
        self.emb_dim = emb_dim

    def forward(self, graph_features, external_features):
        combined_features = torch.cat([graph_features, external_features], dim=-1)
        attn_weights = self.attention(combined_features)
        fused_features = attn_weights * graph_features + (1 - attn_weights) * external_features

        return fused_features
class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="attention",
                 descriptor_dim=1781, external_feature_dim=10):
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.descriptor_dim=descriptor_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.external_feature_dim = external_feature_dim
        self.external_feature_transform = nn.Linear(external_feature_dim, emb_dim)
        self.fusion_linear = nn.Linear(2*emb_dim, emb_dim)

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

        self.NN_descriptor = nn.Sequential(nn.Linear(self.descriptor_dim, self.emb_dim),
                                           nn.Sigmoid(),
                                           nn.Linear(self.emb_dim, self.emb_dim))
        self.sigmoid = nn.Sigmoid()
        self.attention_fusion = AttentionFusion(emb_dim)
        self.transfer_h_graph = nn.Linear(self.emb_dim, 2)


    def forward(self, batched_atom_bond,batched_bond_angle):
        h_node,h_node_ba= self.gnn_node(batched_atom_bond,batched_bond_angle)
        h_graph = self.pool(h_node, batched_atom_bond.batch)
        output = self.graph_pred_linear(h_graph)
        if self.training:
            return output,h_graph
        else:
            return torch.clamp(output, min=0, max=1e8),h_graph

def parse_args():
    parser = argparse.ArgumentParser()
    # --------------parameters for path-----------------------
    parser.add_argument('--clean_data_path', type=str, default='../Data/Clean_database.csv')
    parser.add_argument('--article_figure_path', type=str, default='../Article_figure_path/')
    parser.add_argument('--dft_figure_path', type=str, default='../DFT/')
    parser.add_argument('--Whether_predict', type=bool, default=False)
    parser.add_argument('--shuffle_seed', type=int, default=520)
    parser.add_argument('--known_data_path', type=str, default='../Data/Known.xlsx')
    parser.add_argument('--unknown_data_path', type=str, default='../Data/unKnown.xlsx')
    parser.add_argument('--known_descriptor_path', type=str, default='../Data/Known_Descriptor.xlsx')
    parser.add_argument('--unknown_descriptor_path', type=str, default='../Data/Waiting_pre.xlsx')
    parser.add_argument('--recommend_path', type=str, default='../Data/Recommend.csv')
    parser.add_argument('--recommend_outcome', type=str, default='../Outcome/Recommend/')

    # --------------parameters for ML train and test-----------------------
    parser.add_argument('--use_model', type=str, default='XGB')
    parser.add_argument('--ML_test_ratio', type=float, default=0.1)
    parser.add_argument('--ML_features', nargs='+', type=str, default=['TPSA', 'HBA', 'HBD', 'NROTB', 'MW', 'LogP','Initial_T','Final_T','Heating_rate','Ret_time'])
    parser.add_argument('--ML_label', nargs='+', type=str, default=['Peak_time'])
    parser.add_argument('--ML_seeds',  type=int, default=520)
    parser.add_argument('--save_model_file', type=str, default='../Outcome/ML_model/')
    parser.add_argument('--image_output', type=str, default='../Outcome/image_output/')
    parser.add_argument('--Whether_30min', type=bool, default=False)

    # --------------parameters for XGB-----------------------
    parser.add_argument('--XGB_estimators', type=int, default=300)
    parser.add_argument('--XGB_depth_Regress', type=int, default=7)
    parser.add_argument('--XGB_eta_Regress', type=float, default=0.02)
    parser.add_argument('--XGB_subsample_Regress', type=float, default=0.6)

    # --------------parameters for LGB-----------------------
    parser.add_argument('--LGB_estimators', type=int, default=300)
    parser.add_argument('--LGB_num_leaves_Regress', type=int, default=50)
    parser.add_argument('--LGB_learning_rate_Regress', type=float, default=0.01)
    parser.add_argument('--LGB_max_depth_Regress', type=int, default=8)
    parser.add_argument('--LGB_min_child_samples_Regress', type=int, default=5)

    # --------------parameters for ANN-----------------------
    parser.add_argument('--ann_valid_ratio', type=float, default=0.12)
    parser.add_argument('--NN_num_layer', type=int, default=1)
    parser.add_argument('--NN_hidden_neuron', type=int, default=64)
    parser.add_argument('--BN_model', type=str, default='post')
    parser.add_argument('--Normalize_momentum', type=float, default=0.1)
    parser.add_argument('--act_fun', type=str, default='leaky_relu')
    parser.add_argument('--init_method', type=str, default='Xavier')
    parser.add_argument('--max_iteration', type=int, default=1000)

    # --------------parameters for GNN-----------------------
    parser.add_argument('--GNN_mode', type=str, default='Train')
    parser.add_argument('--Whether_sequence', type=bool, default=True)
    parser.add_argument('--Fig2c_Similarity', type=bool, default=False)
    parser.add_argument('--similarity_threshold', type=float, default=0.8)
    parser.add_argument('--Fig2c_number', type=bool, default=False)
    parser.add_argument('--Fig2c_noise', type=bool, default=False)
    parser.add_argument('--noise_level', type=float, default=0.8)

    parser.add_argument('--Enbedding_mode', type=str, default='Post')
    parser.add_argument('--gnn_train_ratio', type=float, default=0.8)
    parser.add_argument('--gnn_valid_ratio', type=float, default=0.1)
    parser.add_argument('--gnn_test_ratio', type=float, default=0.1)
    parser.add_argument('--known_3D_path', type=str, default='../Data/known_3D_mol_1012')
    parser.add_argument('--unknown_3D_path', type=str, default='../Data/unknown_3D_mol_1012')
    parser.add_argument('--recommend_3D_path', type=str, default='../Data/recommend_3D_mol_1012')
    parser.add_argument('--Train_model_path', type=str, default='../Outcome/GNN_seq240306_230c')
    parser.add_argument('--Test_model_path', type=str, default='/model_save_600.pth')
    parser.add_argument('--predict_model_path', type=str, default='/model_save_600.pth')
    parser.add_argument('--recommend_model_path', type=str, default='/model_save_600.pth')
    parser.add_argument('--Re_smiles_A', type=str, default='O=CC=CC1=CC=CC=C1')
    parser.add_argument('--Re_smiles_B', type=str, default='O=C/C=C/C1=CC=CC=C1')
    parser.add_argument('--Outcome_graph_path', type=str, default='../Outcome/GNN_seq240306_230c/Test_Graph')

    parser.add_argument('--task_name', type=str, default='GINGraphPooling')
    parser.add_argument('--num_task', type=int, default=3)
    parser.add_argument('--num_iterations', type=int, default=1500)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--graph_pooling', type=str, default='sum')
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--gnn_batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default="dataset")

    # --------------parameters for symbolic regression-----------------------
    parser.add_argument('--if_sr', type=bool, default=False)

    args = parser.parse_args()

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config

class Dataset_process():
    '''
    For processing the data and split the dataset
    '''
    def __init__(self,config):
        self.Whether_predict = config.Whether_predict
        self.shuffle_seed = config.shuffle_seed
        self.known_data_path = config.known_data_path
        self.unknown_data_path = config.unknown_data_path
        self.known_descriptor_path = config.known_descriptor_path
        self.unknown_descriptor_path = config.unknown_descriptor_path
        self.ML_test_ratio = config.ML_test_ratio
        self.ML_features = config.ML_features
        self.ML_label = config.ML_label
        self.known_3D_path = config.known_3D_path
        self.unknown_3D_path = config.unknown_3D_path
        self.gnn_train_ratio = config.gnn_train_ratio
        self.gnn_valid_ratio = config.gnn_valid_ratio
        self.gnn_test_ratio = config.gnn_test_ratio
        self.gnn_batch_size = config.gnn_batch_size
        self.num_workers = config.num_workers
        self.clean_data_path = config.clean_data_path
        self.Enbedding_mode = config.Enbedding_mode
        self.Whether_sequence = config.Whether_sequence
        self.Whether_30min = config.Whether_30min
        self.Fig2c_Similarity = config.Fig2c_Similarity
        self.similarity_threshold = config.similarity_threshold
        self.Fig2c_number = config.Fig2c_number
        self.Fig2c_noise = config.Fig2c_noise
        self.noise_level  = config.noise_level
        self.Re_smiles_A = config.Re_smiles_A
        self.Re_smiles_B = config.Re_smiles_B
        self.recommend_3D_path = config.recommend_3D_path
        self.GNN_mode = config.GNN_mode

    def process_GC_data(self):
        GC = pd.read_csv(self.clean_data_path)
        new_df = GC[['Area', 'Height', 'Initial_T', 'Final_T', 'Heating_rate', 'Ret_time',
                     'Smiles', 'Peak_time']]
        new_df.to_excel(self.known_data_path, index=False, engine='openpyxl')

    def process_dataframe(self,df):
        random_seed = 520
        smiles_list = df['Smiles'].tolist()
        tpsa, hba, hbd, nrotb, mw, logp = [], [], [], [], [], []
        maccs_columns = [f"MACCS_bit_{i}" for i in range(166)]  # MACCS总共有166位
        for column in maccs_columns:
            df[column] = 0
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                tpsa.append(Descriptors.TPSA(mol))
                hba.append(Descriptors.NumHAcceptors(mol))
                hbd.append(Descriptors.NumHDonors(mol))
                nrotb.append(Descriptors.NumRotatableBonds(mol))
                mw.append(Descriptors.MolWt(mol))
                logp.append(Descriptors.MolLogP(mol))
                maccs_key = MACCSkeys.GenMACCSKeys(mol)
                for i, bit in enumerate(maccs_key):
                    df.at[smiles_list.index(smiles), f"MACCS_bit_{i}"] = int(bit)
            else:
                tpsa.append(None)
                hba.append(None)
                hbd.append(None)
                nrotb.append(None)
                mw.append(None)
                logp.append(None)
        df['TPSA'] = tpsa
        df['HBA'] = hba
        df['HBD'] = hbd
        df['NROTB'] = nrotb
        df['MW'] = mw
        df['LogP'] = logp
        def calculate_temperatures(row):
            initial_hold_time = row['Ret_time']
            initial_temp = row['Initial_T']
            heating_rate = row['Heating_rate']
            final_temp = row['Final_T']
            temperatures = [initial_temp] * 32
            current_temp = initial_temp
            for minute in range(int(initial_hold_time), 32):
                if current_temp < final_temp:
                    current_temp = min(current_temp + heating_rate, final_temp)
                temperatures[minute] = current_temp
            return temperatures
        temperature_columns = df.apply(calculate_temperatures, axis=1)
        expanded_df = pd.concat([df, pd.DataFrame(temperature_columns.tolist(), index=df.index)], axis=1)
        expanded_df.columns = list(df.columns) + [f'min{i + 1} temperature' for i in range(32)]
        return expanded_df

    def Get_data_file(self):
        known = pd.read_excel(self.known_data_path)
        random_seed = self.shuffle_seed
        known.insert(0, 'Index', known.index)
        known = known.sample(frac=1, random_state=random_seed)
        known_descriptor = self.process_dataframe(known)
        known_descriptor.to_excel(self.known_descriptor_path, index=False)

        if self.Whether_predict == True:
            unknown = pd.read_excel(self.unknown_data_path)
            unknown.insert(0, 'Index', unknown.index)
            unknown_descriptor = self.process_dataframe(unknown)
            unknown_descriptor.to_excel(self.unknown_descriptor_path, index=False)
            return known_descriptor, unknown_descriptor
        else:
            return known_descriptor

    def make_ml_dataset(self):
        data = pd.read_excel(self.known_descriptor_path)
        feature_columns = self.ML_features
        label_columns = self.ML_label
        if self.Whether_30min == False:
            X = data[feature_columns]
        else:
            col = ['TPSA', 'HBA', 'HBD', 'NROTB', 'MW', 'LogP'] + [f'min{i + 1} temperature' for i in range(32)]
            X = data[col]
        y = data[label_columns]
        test_ratio = self.ML_test_ratio
        test_size = int(len(data) * test_ratio)
        train_size = len(data) - test_size
        x_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        x_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        if self.Whether_predict == True:
            data_search = pd.read_excel(self.unknown_descriptor_path)
            x_search = data_search[feature_columns]
            return x_train,y_train,x_test,y_test,x_search
        else:
            return x_train, y_train, x_test, y_test

    def prepare_gnn_3D(self):
        df_known = pd.read_excel(self.known_descriptor_path)
        smiles = df_known['Smiles'].values
        known_3D_path = self.known_3D_path
        if not os.path.exists(known_3D_path):
            os.makedirs(known_3D_path)
        bad_Conformation = save_3D_mol(smiles, known_3D_path)
        np.save(known_3D_path+'/bad_Conformation.npy', np.array(bad_Conformation))
        save_dataset(smiles, known_3D_path, known_3D_path+'/Graph_dataset',known_3D_path+'/Descriptors', bad_Conformation)

        if self.Whether_predict == True:
            df_unknown = pd.read_excel(self.unknown_descriptor_path)
            smiles_unknown = df_unknown['Smiles'].values
            unknown_3D_path = self.unknown_3D_path
            if not os.path.exists(unknown_3D_path):
                os.makedirs(unknown_3D_path)
            bad_Conformation_unknown = save_3D_mol(smiles_unknown, unknown_3D_path)
            np.save(unknown_3D_path + '/bad_Conformation.npy', np.array(bad_Conformation_unknown))
            save_dataset(smiles_unknown, unknown_3D_path, unknown_3D_path + '/Graph_dataset', unknown_3D_path + '/Descriptors', bad_Conformation_unknown)

    def Construct_dataset(self, dataset, data_index, rt, route, ini_tem, last_tem, up_ratio , retion_time):
        graph_atom_bond = []
        graph_bond_angle = []
        big_index = []
        all_descriptor = np.load(route + '/Descriptors.npy')
        for i in range(len(dataset)):
            data = dataset[i]
            atom_feature = []
            bond_feature = []
            for name in atom_id_names:
                atom_feature.append(data[name])
            for name in bond_id_names:
                bond_feature.append(data[name])
            atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
            bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
            bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
            bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
            y = torch.Tensor([float(rt[i])])
            edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
            bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
            data_index_int = torch.from_numpy(np.array(data_index[i])).to(torch.int64)
            TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820] / 100
            RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
            RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
            MDEC = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
            MATS = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]

            bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
            if self.Enbedding_mode == 'Pre':
                chushi = torch.ones([bond_feature.shape[0]]) * ini_tem[i]
                motai = torch.ones([bond_feature.shape[0]]) * last_tem[i]
                sulv = torch.ones([bond_feature.shape[0]]) * up_ratio[i]
                baoliu = torch.ones([bond_feature.shape[0]]) * retion_time[i]
                bond_feature = torch.cat([bond_feature, chushi.reshape(-1, 1), motai.reshape(-1, 1),sulv.reshape(-1, 1),baoliu.reshape(-1, 1)], dim=1)   #拼接

            bond_angle_feature = bond_angle_feature.reshape(-1, 1)
            bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)

            if y[0] > 60:
                big_index.append(i)
                continue
            data_atom_bond = Data(atom_feature, edge_index, bond_feature, y, data_index=data_index_int)
            data_bond_angle = Data(edge_index=bond_index, edge_attr=bond_angle_feature)
            graph_atom_bond.append(data_atom_bond)
            graph_bond_angle.append(data_bond_angle)
        return graph_atom_bond, graph_bond_angle, big_index

    def generate_time_temp_sequence(self, chushi, motai, sulv, baoliu, indices, max_length):
        sequences = []
        for i in indices:
            sequence = []
            time = 0
            temp = chushi[i]
            while time < baoliu[i]:
                sequence.append([temp])
                time += 1
            while temp < motai[i] and len(sequence) < max_length:
                temp += sulv[i]
                sequence.append([temp])
            while len(sequence) < max_length:
                sequence.append([motai[i]])
            sequences.append(torch.tensor(sequence[:max_length], dtype=torch.float32))
        return sequences

    def split_dataset_by_smiles(self,df, ratio):
        df['Molecule'] = df['Smiles'].apply(Chem.MolFromSmiles)
        df['Fingerprint'] = df['Molecule'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
        train_set, valid_set, test_set = set(), set(), set()
        for idx, fp in enumerate(df['Fingerprint']):
            similarities = [DataStructs.TanimotoSimilarity(fp, train_fp) for train_fp in
                            df.loc[train_set, 'Fingerprint']]
            if len(similarities) == 0 or max(similarities) < ratio:
                train_set.add(idx)
            else:
                if np.random.rand() < 0.5:
                    valid_set.add(idx)
                else:
                    test_set.add(idx)
        print(len(list(train_set)))
        print(len(list(valid_set)))
        print(len(list(test_set)))
        return list(train_set), list(valid_set), list(test_set)


    def add_gaussian_noise(self, data, noise_level):
        if noise_level < 0 or noise_level > 1:
            raise ValueError("Noise level must be between 0 and 1")

        std_dev = np.std(data)
        noise = np.random.normal(0, noise_level * std_dev, data.shape)
        return data + noise

    def save_recommend_dataset(self, charity_smile, mol_save_dir, charity_name, moder_name, bad_conformer ,num):
        dataset = []
        dataset_mord = []
        pbar = tqdm(charity_smile)
        index = 0
        for smile in pbar:
            if index in bad_conformer:
                index += 1
                continue
            mol = Chem.MolFromMolFile(f"{mol_save_dir}/3D_mol_{index}.mol")
            descriptor = mord(mol)
            data = mol_to_geognn_graph_data_MMFF3d(mol)
            dataset.append(data)
            dataset_mord.append(descriptor)
            break
        dataset_total = []
        for i in range(num):
            dataset_total.append(dataset[0])

        dataset_mord_total = []
        for i in range(num):
            dataset_mord_total.append(dataset_mord[0])

        dataset_mord_total = np.array(dataset_mord_total)
        np.save(f"{charity_name}.npy", dataset_total, allow_pickle=True)  # 保留图数据
        np.save(f'{moder_name}.npy', dataset_mord_total)  # 保留描述符数据

    def save_3D_recommend_mol(self,all_smile, mol_save_dir):
        index = 0
        bad_conformer = []
        pbar = tqdm(all_smile)
        try:
            os.makedirs(f'{mol_save_dir}')
        except OSError:
            pass
        for smiles in pbar:
            try:
                obtain_3D_mol(smiles, f'{mol_save_dir}/3D_mol_{index}')
            except ValueError:
                bad_conformer.append(index)
                index += 1
                continue
            index += 1
            break
        return bad_conformer  # 保存不好的构象的index

    def predictdataframe_to_sequences(self,df, max_length):
        """"""
        sequences = []
        # Assuming df has columns like 'min1 temperature', 'min2 temperature', ..., 'min30 temperature'
        temperature_columns = [f'min{i + 1} temperature' for i in range(30)]

        for index, row in df.iterrows():
            # Extract temperature values for this row
            temperatures = row[temperature_columns].tolist()

            # If the length of the sequence is less than max_length, extend it
            while len(temperatures) < max_length:
                temperatures.append(temperatures[-1])  # Repeat the last temperature value

            # Trim the sequence if it's longer than max_length
            sequence = temperatures[:max_length]

            # Convert the sequence to a tensor and add to the list
            sequences.append(torch.tensor(sequence, dtype=torch.float32).view(max_length, 1))

        return sequences

    def split_dataset_by_smiles_type(self,df,ratio):
        np.random.seed(3407)
        unique_smiles = df['Smiles'].unique()
        train_size = int(len(unique_smiles) * ratio)
        train_smiles = np.random.choice(unique_smiles, size=train_size, replace=False)
        train_index = df[df['Smiles'].isin(train_smiles)].index
        remaining_df = df[~df['Smiles'].isin(train_smiles)]
        remaining_df_shuffled = remaining_df.sample(frac=1, random_state=seed)
        valid_test_split = int(len(remaining_df_shuffled) / 2)
        valid_index = remaining_df_shuffled.iloc[:valid_test_split].index
        test_index = remaining_df_shuffled.iloc[valid_test_split:].index

        return train_index, valid_index, test_index

    def make_gnn_dataset(self):
        ACA = pd.read_excel(self.known_descriptor_path)
        bad_index = np.load(self.known_3D_path+'/bad_Conformation.npy')
        ACA = ACA.drop(bad_index)
        smiles = ACA['Smiles'].values
        y = ACA[self.ML_label].values
        chushi = ACA['Initial_T'].values
        motai = ACA['Final_T'].values
        sulv = ACA['Heating_rate'].values
        baoliu = ACA['Ret_time'].values
        if self.Whether_30min == False:
            exp_total = np.column_stack((chushi, motai, sulv, baoliu))
        else:
            exp_total = ACA[[f'min{i + 1} temperature' for i in range(30)]].values
        graph_dataset = np.load(self.known_3D_path+'/Graph_dataset.npy', allow_pickle=True).tolist()
        index_aca = ACA['Index'].values
        dataset_graph_atom_bond, dataset_graph_bond_angle, big_index = self.Construct_dataset(graph_dataset, index_aca, y, self.known_3D_path,chushi, motai,sulv,baoliu)  #生成两张图：atom-bond和 bond-angle图
        total_num = len(dataset_graph_atom_bond)
        print('Known data num:', total_num)
        train_ratio = self.gnn_train_ratio
        validate_ratio = self.gnn_valid_ratio
        test_ratio = self.gnn_test_ratio
        data_array = np.arange(0, total_num, 1)
        np.random.seed(520)
        np.random.shuffle(data_array)
        torch.random.manual_seed(520)
        train_data_atom_bond = []
        valid_data_atom_bond = []
        test_data_atom_bond = []
        train_data_bond_angle = []
        valid_data_bond_angle = []
        test_data_bond_angle = []
        train_num = int(len(data_array) * train_ratio)
        test_num = int(len(data_array) * test_ratio)
        val_num = int(len(data_array) * validate_ratio)
        if self.Fig2c_noise == True:
            exp_total = self.add_gaussian_noise(exp_total, noise_level=self.noise_level)
            chushi = self.add_gaussian_noise(chushi, noise_level=self.noise_level)
            motai = self.add_gaussian_noise(motai, noise_level=self.noise_level)
            sulv = self.add_gaussian_noise(sulv, noise_level=self.noise_level)
            baoliu = self.add_gaussian_noise(baoliu, noise_level=self.noise_level)
            train_index = data_array[0:train_num]
            valid_index = data_array[train_num:train_num + val_num]
            test_index = data_array[total_num - test_num:]
        elif self.Fig2c_Similarity == True:
            train_index, valid_index, test_index = self.split_dataset_by_smiles_type(ACA, ratio=self.similarity_threshold)
        else:
            train_index = data_array[0:train_num]
            valid_index = data_array[train_num:train_num + val_num]
            test_index = data_array[total_num - test_num:]
        train_exp = torch.tensor(exp_total[train_index], dtype=torch.float32)
        valid_exp = torch.tensor(exp_total[valid_index], dtype=torch.float32)
        test_exp = torch.tensor(exp_total[test_index], dtype=torch.float32)

        for i in test_index:
            test_data_atom_bond.append(dataset_graph_atom_bond[i])
            test_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in valid_index:
            valid_data_atom_bond.append(dataset_graph_atom_bond[i])
            valid_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in train_index:
            train_data_atom_bond.append(dataset_graph_atom_bond[i])
            train_data_bond_angle.append(dataset_graph_bond_angle[i])
        train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,
                                            num_workers=self.num_workers)
        valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,
                                            num_workers=self.num_workers)
        test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,
                                           num_workers=self.num_workers)
        train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,
                                             num_workers=self.num_workers)
        valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,
                                             num_workers=self.num_workers)
        test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,
                                            num_workers=self.num_workers)

        train_exp_loader = DataLoader(train_exp, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
        valid_exp_loader = DataLoader(valid_exp, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
        test_exp_loader = DataLoader(test_exp, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)

        if self.Whether_sequence == False:
            return train_loader_atom_bond,valid_loader_atom_bond,test_loader_atom_bond,train_loader_bond_angle,\
                valid_loader_bond_angle,test_loader_bond_angle,train_exp_loader,valid_exp_loader,test_exp_loader
        else:
            max_sequence_length = 32  # Define a fixed sequence length
            train_seq = self.generate_time_temp_sequence(chushi, motai, sulv, baoliu, train_index, max_sequence_length)
            valid_seq = self.generate_time_temp_sequence(chushi, motai, sulv, baoliu, valid_index, max_sequence_length)
            test_seq = self.generate_time_temp_sequence(chushi, motai, sulv, baoliu, test_index, max_sequence_length)
            train_seq_loader = DataLoader(train_seq, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
            valid_seq_loader = DataLoader(valid_seq, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
            test_seq_loader = DataLoader(test_seq, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)

            return train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond, \
                train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle, \
                train_seq_loader, valid_seq_loader, test_seq_loader

    def copy_file_in_path(self,file_path, target_path, n):
        base_name, extension = os.path.splitext(os.path.basename(file_path))
        base_name = "_".join(base_name.split("_")[:-1])

        for i in range(1, n):
            new_file_name = f"{base_name}_{i}{extension}"
            new_file_path = os.path.join(target_path, new_file_name)
            shutil.copy(file_path, new_file_path)

    def make_gnn_prediction_dataset(self,df):
        df_unknown = pd.read_excel(df, header=None)
        column_names = ['Smiles'] + [f'min{i + 1} temperature' for i in range(30)] + ['True_RT']
        if len(df_unknown.columns) == len(column_names):
            df_unknown.columns = column_names
        else:
            raise ValueError("The number of columns in the DataFrame does not match the number of provided column names.")
        smiles_unknown = df_unknown['Smiles'].values
        unknown_3D_path = self.unknown_3D_path
        if not os.path.exists(unknown_3D_path):
            os.makedirs(unknown_3D_path)

        if self.GNN_mode == 'Pre':
            bad_Conformation_unknown = save_3D_mol(smiles_unknown, unknown_3D_path)
            np.save(unknown_3D_path + '/bad_Conformation.npy', np.array(bad_Conformation_unknown))
            save_dataset(smiles_unknown, unknown_3D_path, unknown_3D_path + '/Graph_dataset', unknown_3D_path + '/Descriptors', bad_Conformation_unknown)
        else:
            bad_Conformation_unknown = self.save_3D_recommend_mol(smiles_unknown, unknown_3D_path)
            np.save(unknown_3D_path + '/bad_Conformation.npy', np.array(bad_Conformation_unknown))
            self.copy_file_in_path(unknown_3D_path + '/3D_mol_0.mol', unknown_3D_path, len(smiles_unknown))
            self.save_recommend_dataset(smiles_unknown, unknown_3D_path, unknown_3D_path + '/Graph_dataset',
                                        unknown_3D_path + '/Descriptors', bad_Conformation_unknown, len(smiles_unknown))
        ACA = df_unknown
        bad_index = np.load(self.unknown_3D_path+'/bad_Conformation.npy')
        ACA = ACA.drop(bad_index)
        smiles = ACA['Smiles'].values
        y = ACA['True_RT'].values
        graph_dataset = np.load(self.unknown_3D_path+'/Graph_dataset.npy',allow_pickle=True).tolist()
        index_aca = ACA.index
        chushi = ACA['True_RT'].values
        motai = ACA['True_RT'].values
        sulv = ACA['True_RT'].values
        baoliu = ACA['True_RT'].values

        dataset_graph_atom_bond, dataset_graph_bond_angle, big_index = self.Construct_dataset(graph_dataset, index_aca,
                                                                                              y, self.unknown_3D_path,
                                                                                              chushi, motai, sulv,
                                                                                              baoliu)
        exp_total = ACA[[f'min{i + 1} temperature' for i in range(30)]].values
        pre_exp = torch.tensor(exp_total, dtype=torch.float32)
        pre_exp_loader = DataLoader(pre_exp, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
        pre_loader_atom_bond = DataLoader(dataset_graph_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
        pre_loader_bond_angle = DataLoader(dataset_graph_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
        if self.Whether_sequence == False:
            return pre_loader_atom_bond,pre_loader_bond_angle,pre_exp_loader
        else:
            max_sequence_length = 32  # Define a fixed sequence length
            pre_seq = self.predictdataframe_to_sequences(ACA,max_sequence_length)
            # Creating DataLoader for the time-temperature sequences
            pre_seq_loader = DataLoader(pre_seq, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
            return pre_loader_atom_bond, pre_loader_bond_angle, pre_seq_loader

    def make_gnn_recommend_dataset(self,df):
        df_unknown = pd.read_excel(df, header=None)
        column_names = ['Smiles'] + [f'min{i + 1} temperature' for i in range(30)] + ['True_RT']
        if len(df_unknown.columns) == len(column_names):
            df_unknown.columns = column_names
        else:
            raise ValueError(
                "The number of columns in the DataFrame does not match the number of provided column names.")
        smiles_unknown = df_unknown['Smiles'].values
        unknown_3D_path = self.recommend_3D_path
        if not os.path.exists(unknown_3D_path):
            os.makedirs(unknown_3D_path)

        bad_Conformation_unknown = save_3D_mol(smiles_unknown, unknown_3D_path)
        np.save(unknown_3D_path + '/bad_Conformation.npy', np.array(bad_Conformation_unknown))
        save_dataset(smiles_unknown, unknown_3D_path, unknown_3D_path + '/Graph_dataset',
                     unknown_3D_path + '/Descriptors', bad_Conformation_unknown)

        bad_Conformation_unknown = self.save_3D_recommend_mol(smiles_unknown, unknown_3D_path)
        np.save(unknown_3D_path + '/bad_Conformation.npy', np.array(bad_Conformation_unknown))
        self.copy_file_in_path(unknown_3D_path+'/3D_mol_0.mol',unknown_3D_path,len(smiles_unknown))
        self.save_recommend_dataset(smiles_unknown, unknown_3D_path, unknown_3D_path + '/Graph_dataset',
                     unknown_3D_path + '/Descriptors', bad_Conformation_unknown,len(smiles_unknown))

        ACA = df_unknown
        bad_index = np.load(self.unknown_3D_path + '/bad_Conformation.npy')
        ACA = ACA.drop(bad_index)
        smiles = ACA['Smiles'].values
        y = ACA['True_RT'].values
        graph_dataset = np.load(self.unknown_3D_path + '/Graph_dataset.npy', allow_pickle=True).tolist()
        index_aca = ACA.index
        chushi = ACA['True_RT'].values
        motai = ACA['True_RT'].values
        sulv = ACA['True_RT'].values
        baoliu = ACA['True_RT'].values

        dataset_graph_atom_bond, dataset_graph_bond_angle, big_index = self.Construct_dataset(graph_dataset, index_aca,
                                                                                              y, unknown_3D_path,
                                                                                              chushi, motai, sulv,
                                                                                              baoliu)
        exp_total = ACA[[f'min{i + 1} temperature' for i in range(30)]].values
        pre_exp = torch.tensor(exp_total, dtype=torch.float32)
        pre_exp_loader = DataLoader(pre_exp, batch_size=self.gnn_batch_size, shuffle=False,
                                    num_workers=self.num_workers)
        pre_loader_atom_bond = DataLoader(dataset_graph_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,
                                          num_workers=self.num_workers)
        pre_loader_bond_angle = DataLoader(dataset_graph_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,
                                           num_workers=self.num_workers)
        if self.Whether_sequence == False:
            return pre_loader_atom_bond, pre_loader_bond_angle, pre_exp_loader
        else:
            max_sequence_length = 32  # Define a fixed sequence length
            pre_seq = self.predictdataframe_to_sequences(ACA, max_sequence_length)
            # Creating DataLoader for the time-temperature sequences
            pre_seq_loader = DataLoader(pre_seq, batch_size=self.gnn_batch_size, shuffle=False,
                                        num_workers=self.num_workers)
            return pre_loader_atom_bond, pre_loader_bond_angle, pre_seq_loader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AttentionLayer(nn.Module):
    def __init__(self, d_model, seq_len):
        super(AttentionLayer, self).__init__()
        self.seq_len = seq_len
        self.attention_weights = nn.Linear(d_model, seq_len)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        attention_scores = self.attention_weights(x).mean(dim=-1)
        attention_weights = F.softmax(attention_scores, dim=0)
        weighted_x = x * attention_weights.unsqueeze(-1)
        return weighted_x.sum(dim=0)

class ExperimentConditionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExperimentConditionModule, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.layer(x)

class TimeTempTransformerModule(nn.Module):
    def __init__(self, seq_len=32, d_model=32, num_layers=1, output_dim=128, dropout=0.1):
        super(TimeTempTransformerModule, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.feature_extractor = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, d_model)
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional = True
        )
        self.decoder = nn.Linear(d_model*2, output_dim)

    def forward(self, x):
        x = x.reshape(-1, self.seq_len, 1)
        x = self.feature_extractor(x)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x

class MultiSeqModel(nn.Module):
    def __init__(self, graph_nn_params, transfomer_nn_params, num_tasks):
        super(MultiSeqModel, self).__init__()
        self.graph_module = GINGraphPooling(**graph_nn_params)
        self.exp_condition_module = TimeTempTransformerModule(**transfomer_nn_params,output_dim = graph_nn_params['emb_dim'])
        self.attention_weights = nn.Parameter(torch.rand(2), requires_grad=True)
        self.fusion_layer_1 = nn.Linear(graph_nn_params['emb_dim'], 256)
        self.dropout = nn.Dropout(0.5)
        self.norm1 = nn.LayerNorm(256)
        self.fusion_layer_2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.final_layer = nn.Linear(128, num_tasks)
        self.residual1 = nn.Linear(graph_nn_params['emb_dim'], 256)
        self.residual2 = nn.Linear(256, 128)

    def forward(self, batched_atom_bond, batched_bond_angle, exp_conditions):

        _, graph_representation = self.graph_module(batched_atom_bond, batched_bond_angle)
        exp_condition_representation = self.exp_condition_module(exp_conditions)
        attention_probs = F.softmax(self.attention_weights, dim=0)
        weighted_representation = attention_probs[0] * graph_representation + attention_probs[1] * exp_condition_representation
        x = F.leaky_relu(self.fusion_layer_1(weighted_representation) + self.residual1(weighted_representation))
        x = self.dropout(self.norm1(x))
        x = F.leaky_relu(self.fusion_layer_2(x) + self.residual2(x))
        x = self.norm2(x)
        output = self.final_layer(x)
        return output

class MultiModalModel(nn.Module):
    def __init__(self, graph_nn_params, exp_condition_dim, num_tasks):
        super(MultiModalModel, self).__init__()
        self.graph_module = GINGraphPooling(**graph_nn_params)
        self.exp_condition_module = ExperimentConditionModule(exp_condition_dim, graph_nn_params['emb_dim'])
        self.attention_weights = nn.Parameter(torch.rand(2), requires_grad=True)
        self.fusion_layer_1 = nn.Linear(graph_nn_params['emb_dim'], 256)
        self.dropout = nn.Dropout(0.5)
        self.norm1 = nn.LayerNorm(256)
        self.fusion_layer_2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.final_layer = nn.Linear(128, num_tasks)
        self.residual1 = nn.Linear(graph_nn_params['emb_dim'], 256)
        self.residual2 = nn.Linear(256, 128)

    def forward(self, batched_atom_bond, batched_bond_angle, exp_conditions):
        _ , graph_representation = self.graph_module(batched_atom_bond, batched_bond_angle)
        exp_condition_representation = self.exp_condition_module(exp_conditions)
        attention_probs = F.softmax(self.attention_weights, dim=0)
        weighted_representation = attention_probs[0] * graph_representation + attention_probs[1] * exp_condition_representation
        x = F.leaky_relu(self.fusion_layer_1(weighted_representation) + self.residual1(weighted_representation))
        x = self.dropout(self.norm1(x))
        x = F.leaky_relu(self.fusion_layer_2(x) + self.residual2(x))
        x = self.norm2(x)
        output = self.final_layer(x)
        return output

class GNN_process():
    def __init__(self,config):
        self.GNN_mode = config.GNN_mode
        self.num_task = config.num_task
        self.num_layers = config.num_layers
        self.emb_dim = config.emb_dim
        self.drop_ratio = config.drop_ratio
        self.graph_pooling = config.graph_pooling
        self.weight_decay = config.weight_decay
        self.Train_model_path = config.Train_model_path
        self.num_iterations = config.num_iterations
        self.Test_model_path = config.Test_model_path
        self.gnn_train_ratio = config.gnn_train_ratio
        self.gnn_valid_ratio = config.gnn_valid_ratio
        self.gnn_test_ratio = config.gnn_test_ratio
        self.Outcome_graph_path = config.Outcome_graph_path
        self.predict_model_path = config.predict_model_path
        self.unknown_descriptor_path = config.unknown_descriptor_path
        self.Whether_sequence = config.Whether_sequence
        self.Fig2c_Similarity = config.Fig2c_Similarity
        self.similarity_threshold = config.similarity_threshold
        self.Fig2c_number = config.Fig2c_number
        self.Fig2c_noise = config.Fig2c_noise
        self.noise_level = config.noise_level
        self.recommend_model_path = config.recommend_model_path
        self.Re_smiles_A = config.Re_smiles_A
        self.Re_smiles_B = config.Re_smiles_B
        self.recommend_path = config.recommend_path
        self.recommend_outcome = config.recommend_outcome
        self.recommend_3D_path = config.recommend_3D_path

    def train_gnn(self,model, device, loader_atom_bond, loader_bond_angle, optimizer,train_exp_loader):
        model.train()
        loss_accum = 0
        for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle, train_exp_loader)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            train_exp = batch[2]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            train_exp = train_exp.to(device)
            pred = model(batch_atom_bond, batch_bond_angle,train_exp)
            true = batch_atom_bond.y
            optimizer.zero_grad()
            loss = q_loss(0.1, true, pred[:, 0]) + torch.mean((true - pred[:, 1]) ** 2) + q_loss(0.9, true, pred[:, 2]) \
                   + torch.mean(torch.relu(pred[:, 0] - pred[:, 1])) + torch.mean(
                torch.relu(pred[:, 1] - pred[:, 2])) + torch.mean(torch.relu(2 - pred))
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        return loss_accum / (step + 1)

    def te_gnn(self,model, device, loader_atom_bond, loader_bond_angle,test_exp_loader):
        model.eval()
        y_pred = []
        y_true = []
        y_pred_10 = []
        y_pred_90 = []
        with torch.no_grad():
            for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle,test_exp_loader)):
                batch_atom_bond = batch[0]
                batch_bond_angle = batch[1]
                test_exp = batch[2]
                batch_atom_bond = batch_atom_bond.to(device)
                batch_bond_angle = batch_bond_angle.to(device)
                test_exp = test_exp.to(device)
                pred = model(batch_atom_bond, batch_bond_angle,test_exp)
                y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1, ))
                y_pred.append(pred[:, 1].detach().cpu())
                y_pred_10.append(pred[:, 0].detach().cpu())
                y_pred_90.append(pred[:, 2].detach().cpu())
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            y_pred_10 = torch.cat(y_pred_10, dim=0)
            y_pred_90 = torch.cat(y_pred_90, dim=0)
        R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_pred.mean()) ** 2).sum())
        test_mae = torch.mean((y_true - y_pred) ** 2)
        print(R_square)
        return y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90

    def eval(self, model, device, loader_atom_bond, loader_bond_angle,eval_exp_loader):
        model.eval()
        y_true = []
        y_pred = []
        y_pred_10 = []
        y_pred_90 = []

        with torch.no_grad():
            for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle,eval_exp_loader)):
                batch_atom_bond = batch[0]
                batch_bond_angle = batch[1]
                eval_exp = batch[2]
                batch_atom_bond = batch_atom_bond.to(device)
                batch_bond_angle = batch_bond_angle.to(device)
                eval_exp = eval_exp.to(device)
                pred = model(batch_atom_bond, batch_bond_angle,eval_exp)
                y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
                y_pred.append(pred[:, 1].detach().cpu())
                y_pred_10.append(pred[:, 0].detach().cpu())
                y_pred_90.append(pred[:, 2].detach().cpu())
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred_10 = torch.cat(y_pred_10, dim=0)
        y_pred_90 = torch.cat(y_pred_90, dim=0)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return torch.mean((y_true - y_pred) ** 2).data.numpy()

    def calculate_temperatures(self,row):
        temperatures = [row['Initial Temperature']]
        current_temp = row['Initial Temperature']
        time = 0
        while time < 30:
            if time >= row['Initial Temperature Hold Time']:
                current_temp += row['Heating Rate']
            temperatures.append(min(current_temp, row['Final Temperature']))
            time += 1
        return temperatures[1:]

    def Mode(self):
        if self.GNN_mode == 'Train':
            nn_params = {
                'num_tasks': self.num_task,
                'num_layers': self.num_layers,
                'emb_dim': self.emb_dim,
                'drop_ratio': self.drop_ratio,
                'graph_pooling': self.graph_pooling,
                'descriptor_dim': 1827
            }
            transformer_params = {
                'seq_len': 32,
                'd_model': 64,
                'num_layers': 3,
                'dropout': 0.1
            }

            config = parse_args()
            data = Dataset_process(config)
            data.gnn_train_ratio = self.gnn_train_ratio
            data.gnn_test_ratio = self.gnn_test_ratio
            data.gnn_valid_ratio = self.gnn_valid_ratio
            data.Fig2c_Similarity = self.Fig2c_Similarity
            data.similarity_threshold = self.similarity_threshold
            data.Fig2c_number = self.Fig2c_number
            data.Fig2c_noise = self.Fig2c_noise
            data.noise_level = self.noise_level
            train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond, train_loader_bond_angle, \
                valid_loader_bond_angle, test_loader_bond_angle, train_exp_loader, valid_exp_loader, test_exp_loader = data.make_gnn_dataset()
            criterion_fn = torch.nn.MSELoss()
            exp_condition_dim = 30
            num_tasks = self.num_task
            if self.Whether_sequence == False:
                model = MultiModalModel(graph_nn_params=nn_params, exp_condition_dim = exp_condition_dim,num_tasks = num_tasks).to(device)
            else:
                model = MultiSeqModel(graph_nn_params =nn_params , transfomer_nn_params = transformer_params, num_tasks=num_tasks).to(device)

            num_params = sum(p.numel() for p in model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=self.weight_decay)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
            folder_path = self.Train_model_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            lowest_mae = float('inf')
            lowest_mae_filename = None
            for epoch in tqdm(range(self.num_iterations)):
                train_mae = self.train_gnn(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer,train_exp_loader)
                valid_outcome = []
                if (epoch + 1) % 100 == 0:
                    valid_mae = self.eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle,valid_exp_loader)
                    print(train_mae, valid_mae)
                    valid_outcome.append(valid_mae)
                    current_filename = f'/model_save_{epoch + 1}.pth'
                    torch.save(model.state_dict(), folder_path + current_filename)
                    if valid_mae < lowest_mae:
                        lowest_mae = valid_mae
                        lowest_mae_filename = current_filename
                scheduler.step()
            print(f"The .pth file with the lowest validation MAE ({lowest_mae}) is: {lowest_mae_filename}")
            return lowest_mae_filename

        if self.GNN_mode == 'Test':
            nn_params = {
                'num_tasks': self.num_task,
                'num_layers': self.num_layers,
                'emb_dim': self.emb_dim,
                'drop_ratio': self.drop_ratio,
                'graph_pooling': self.graph_pooling,
                'descriptor_dim': 1827
            }

            transformer_params = {
                'seq_len': 32,
                'd_model': 64,
                'num_layers': 3,
                'dropout': 0.1
            }

            exp_condition_dim = 30
            num_tasks = self.num_task
            if self.Whether_sequence == False:
                model = MultiModalModel(graph_nn_params=nn_params, exp_condition_dim=exp_condition_dim,
                                        num_tasks=num_tasks).to(device)
            else:
                model = MultiSeqModel(graph_nn_params=nn_params, transfomer_nn_params=transformer_params,
                                      num_tasks=num_tasks).to(device)
            model.load_state_dict(torch.load(self.Train_model_path+self.Test_model_path))
            config = parse_args()
            data = Dataset_process(config)
            data.gnn_train_ratio = self.gnn_train_ratio
            data.gnn_test_ratio = self.gnn_test_ratio
            data.gnn_valid_ratio = self.gnn_valid_ratio
            data.Fig2c_Similarity = self.Fig2c_Similarity
            data.similarity_threshold = self.similarity_threshold
            data.Fig2c_number = self.Fig2c_number
            data.Fig2c_noise = self.Fig2c_noise
            data.noise_level = self.noise_level
            train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond, train_loader_bond_angle, \
                valid_loader_bond_angle, test_loader_bond_angle, train_exp_loader, valid_exp_loader, test_exp_loader = data.make_gnn_dataset()
            y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90 = self.te_gnn(model, device, test_loader_atom_bond,test_loader_bond_angle,test_exp_loader)
            y_pred = y_pred.cpu().data.numpy()
            y_true = y_true.cpu().data.numpy()
            y_pred_10 = y_pred_10.cpu().data.numpy()
            y_pred_90 = y_pred_90.cpu().data.numpy()
            in_range = np.sum((y_true >= y_pred_10) & (y_true <= y_pred_90))
            ratio = in_range / len(y_true)
            print('relative_error', np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)))
            print('MAE', np.mean(np.abs(y_true - y_pred) / y_true))
            print('RMSE', np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = np.mean(np.abs(y_true - y_pred) / y_true)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
            print(R_square)
            plt.figure(1, figsize=(3.5, 3.5), dpi=300)
            plt.style.use('ggplot')
            plt.scatter(y_true, y_pred, c='#8983BF', s=15, alpha=0.4)
            plt.plot(np.arange(0, 23), np.arange(0, 23), linewidth=.5, linestyle='--', color='black')
            plt.yticks(np.arange(0, 23, 2), np.arange(0, 23, 2), fontproperties='Arial', size=8)
            plt.xticks(np.arange(0, 23, 2), np.arange(0, 23, 2), fontproperties='Arial', size=8)
            plt.xlabel('Observed data', fontproperties='Arial', size=8)
            plt.ylabel('Predicted data', fontproperties='Arial', size=8)
            plt.text(4, 16, "R2 = {:.5f}".format(R_square), fontsize=10, ha='center')
            plt.text(4, 13, "ratio = {:.4f}".format(ratio), fontsize=10, ha='center')
            folder_path = self.Outcome_graph_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if self.gnn_train_ratio == 0.9:
                plt.title('GNN_18_1_1')
                plt.savefig(folder_path + "/GNN_18_1_1.svg")
            if self.gnn_train_ratio == 0.8:
                plt.title('Multimodal model')
                plt.savefig(folder_path + "/GNN_8_1_1.svg")
            if self.Fig2c_Similarity==False and self.Fig2c_number == False and self.Fig2c_noise == False:
                plt.show()
            return R_square,mae,rmse

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()

        # Layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)  # Output layer

        self.dropout = nn.Dropout(0.2)  # Dropout
        self.activation = nn.LeakyReLU()

        # Residual Connection
        self.residual = nn.Linear(input_dim, 256)

    def forward(self, x):
        identity = self.residual(x)

        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x += identity  # Adding residual connection
        x = self.fc3(x)

        return x

class Model_ML():
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.counter = 0
        self.ML_label = config.ML_label
        self.use_model = config.use_model
        self.ML_seeds = config.ML_seeds
        self.save_model_file = config.save_model_file
        self.XGB_estimators = config.XGB_estimators
        self.LGB_estimators = config.LGB_estimators
        self.XGB_eta_Regress = config.XGB_eta_Regress
        self.XGB_depth_Regress = config.XGB_depth_Regress
        self.XGB_subsample_Regress = config.XGB_subsample_Regress
        self.LGB_learning_rate_Regress = config.LGB_learning_rate_Regress
        self.LGB_max_depth_Regress = config.LGB_max_depth_Regress
        self.LGB_min_child_samples_Regress = config.LGB_min_child_samples_Regress
        self.LGB_num_leaves_Regress = config.LGB_num_leaves_Regress
        self.device = config.device
        self.max_iteration = config.max_iteration
        self.image_output = config.image_output
        self.Whether_30min = config.Whether_30min

    def ensure_directory_exists(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"The directory {directory_path} has been created.")
        else:
            print(f"The directory {directory_path} already exists.")

    def custom_r2(self, y_true, y_pred):
        self.counter += 1
        print('Finished {0}/{1} iterations'.format(self.counter, self.total_iterations))
        return r2_score(y_true, y_pred)

    def train(self, x_train, y_train):
        if self.use_model == 'RF':
            y_cla = y_train[self.ML_label].values.ravel()
            clf = RandomForestRegressor(n_estimators=300, random_state=self.ML_seeds)
            clf.fit(x_train, y_cla)
            y_pre = clf.predict(x_train)
            r2 = r2_score(y_cla, y_pre)
            print('The model is {},the regression mode'.format(self.use_model))
            print('The Train R2 score is {}'.format(r2))
            self.ensure_directory_exists(self.save_model_file)
            dump(clf, self.save_model_file + 'RF_best_model_regression.joblib')
            return clf

        if self.use_model == 'XGB':
            y_cla = y_train[self.ML_label].values.ravel()
            clf = XGBRegressor(n_estimators=self.XGB_estimators, eta=self.XGB_eta_Regress, random_state=self.ML_seeds,
                               max_depth=self.XGB_depth_Regress, subsample=self.XGB_subsample_Regress)
            clf.fit(x_train, y_cla)
            y_pre = clf.predict(x_train)
            r2 = r2_score(y_cla, y_pre)
            print('The model is {},the regression mode'.format(self.use_model))
            print('The Train R2 score is {}'.format(r2))
            self.ensure_directory_exists(self.save_model_file)
            dump(clf, self.save_model_file + 'XGB_best_model_regression.joblib')
            return clf

        if self.use_model == 'LGB':
            y_cla = y_train[self.ML_label].values.ravel()
            clf = LGBMRegressor(n_estimators=self.LGB_estimators, random_state=self.ML_seeds,
                                learning_rate=self.LGB_learning_rate_Regress,
                                max_depth=self.LGB_max_depth_Regress,
                                min_child_samples=self.LGB_min_child_samples_Regress,
                                num_leaves=self.LGB_num_leaves_Regress)
            clf.fit(x_train, y_cla)
            y_pre = clf.predict(x_train)
            r2 = r2_score(y_cla, y_pre)
            print('The model is {},the regression mode'.format(self.use_model))
            print('The Train R2 score is {}'.format(r2))
            self.ensure_directory_exists(self.save_model_file)
            dump(clf, self.save_model_file + 'LGB_best_model_regression.joblib')
            return clf

        if self.use_model == 'ANN':
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

            x_train = torch.tensor(x_train.values, dtype=torch.float32).to(self.device)
            y_train_cla = torch.tensor(y_train[self.ML_label].values.ravel(), dtype=torch.float32).to(self.device)
            train_dataset = TensorDataset(x_train, y_train_cla)
            train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
            for inputs, labels in train_dataloader:
                print(inputs.size())
                break
            x_valid = torch.tensor(x_valid.values, dtype=torch.float32).to(self.device)
            y_valid = torch.tensor(y_valid[self.ML_label].values.ravel(), dtype=torch.float32).to(self.device)
            valid_dataset = TensorDataset(x_valid, y_valid)
            valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False,drop_last=True)

            iteration = self.max_iteration
            model = SimpleNN(x_train.shape[1]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,
                                                          cycle_momentum=False)
            print("Working on: {}".format(self.device))  # Print the current device
            early_stopping_threshold = 1e-3
            early_stopping_counter = 0
            best_val_loss = float('inf')
            best_model_state = copy.deepcopy(model.state_dict())
            for epoch in range(self.max_iteration):
                model.train()
                train_loss = 0.0
                for inputs, labels in train_dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in valid_dataloader:
                        outputs = model(inputs)
                        loss = criterion(outputs.squeeze(), labels)
                        valid_loss += loss.item()
                scheduler.step()
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                if epoch % 10 == 0:
                    print(
                        f'Epoch {epoch}/{self.max_iteration} - Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
            model.load_state_dict(best_model_state)
            self.ensure_directory_exists(self.save_model_file)
            torch.save(model.state_dict(), self.save_model_file + 'ANN_best_model_regression.pth')

    def test(self, x_test, y_test):
        y_cla = y_test[self.ML_label].values.ravel()
        if self.use_model == 'ANN':
            model = SimpleNN(x_test.shape[1]).to(self.device)
            model.load_state_dict(torch.load(self.save_model_file + 'ANN_best_model_regression.pth'))
            model.eval()
            summary(model, input_size=(x_test.shape[1],))
            x = torch.tensor(x_test.values, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                y_pre = model(x).detach().cpu()
        else:
            model = load(self.save_model_file + '{}_best_model_regression.joblib'.format(self.use_model))
            y_pre = model.predict(x_test)
        plt.figure(1, figsize=(3.5, 3.5), dpi=300)
        plt.style.use('ggplot')
        r2 = r2_score(y_cla, y_pre)
        filtered_list1, filtered_list2 = zip(
            *[(item1, item2) for index, (item1, item2) in enumerate(zip(y_cla, y_pre)) if item1 <= 22])
        filtered_list1 = list(filtered_list1)
        filtered_list2 = list(filtered_list2)
        plt.scatter(filtered_list1, filtered_list2, c='#8983BF', s=15, alpha=0.4)
        plt.plot(np.arange(0, 22), np.arange(0, 22), linewidth=.5, linestyle='--', color='black')
        plt.yticks(np.arange(0, 22, 2), np.arange(0, 22, 2), fontproperties='Arial', size=8)
        plt.xticks(np.arange(0, 22, 2), np.arange(0, 22, 2), fontproperties='Arial', size=8)
        plt.xlabel('Peak_time-Observed', fontproperties='Arial', size=8)
        plt.ylabel('Peak_time-Predicted', fontproperties='Arial', size=8)
        plt.text(4, 16, "R2 = {:.5f}".format(r2), fontsize=10, ha='center')
        plt.title('{}'.format(self.use_model))
        print('The model is {},the regression mode'.format(self.use_model))
        print('The test R2 score is {}'.format(r2))
        self.ensure_directory_exists(self.image_output)
        plt.savefig(self.image_output + "{}_231023.svg".format(self.use_model))
        plt.show()
        return r2, y_pre

def main():
    config = parse_args()
    data = Dataset_process(config)
    GNN = GNN_process(config)
    GNN.num_iterations = 1500
    GNN.Mode()

if __name__ == "__main__":
    main()