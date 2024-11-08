from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import math
import torch.nn.functional as F
import torch.nn as nn
from Model.compound_tools import *
from mordred import Calculator, descriptors,is_missing
import random

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
