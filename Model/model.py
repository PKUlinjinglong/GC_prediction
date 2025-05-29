from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn

import copy
from joblib import dump, load
from sklearn.metrics import r2_score
import os
from mordred import Calculator, descriptors,is_missing
import random
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from torchsummary import summary
from torch.utils.data import TensorDataset
from sklearn.ensemble import RandomForestRegressor

from Model.compound_tools import *
from Model.data_process import Dataset_process,q_loss
from Model.GIN import GINGraphPooling
from Model.config import parse_args

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

# Adds sinusoidal positional encodings to input embeddings for Transformer models
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

# Applies simple self-attention over a sequence by computing weighted average
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

# Processes experimental condition vectors using an MLP
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

# Encodes sequential experimental temperature/time data using GRU
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
            bidirectional=True
        )
        self.decoder = nn.Linear(d_model * 2, output_dim)

    def forward(self, x):
        x = x.reshape(-1, self.seq_len, 1)
        x = self.feature_extractor(x)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x

# Combines graph representation and sequential experimental conditions using attention fusion
class MultiSeqModel(nn.Module):
    def __init__(self, graph_nn_params, transfomer_nn_params, num_tasks):
        super(MultiSeqModel, self).__init__()
        self.graph_module = GINGraphPooling(**graph_nn_params)
        self.exp_condition_module = TimeTempTransformerModule(**transfomer_nn_params, output_dim=graph_nn_params['emb_dim'])
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

# Combines graph representation and condition vector using attention fusion
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

# GNN_process handles training, testing, and evaluation of GNN models with experimental conditions
class GNN_process():
    def __init__(self, config):
        # Configuration setup
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

    # Train the GNN model for one epoch
    def train_gnn(self, model, device, loader_atom_bond, loader_bond_angle, optimizer, train_exp_loader):
        model.train()
        loss_accum = 0
        for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle, train_exp_loader)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            train_exp = batch[2]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            train_exp = train_exp.to(device)

            # Forward pass
            pred = model(batch_atom_bond, batch_bond_angle, train_exp)
            true = batch_atom_bond.y

            # Compute custom loss with quantile and ranking constraints
            optimizer.zero_grad()
            loss = q_loss(0.1, true, pred[:, 0]) + \
                   torch.mean((true - pred[:, 1]) ** 2) + \
                   q_loss(0.9, true, pred[:, 2]) + \
                   torch.mean(torch.relu(pred[:, 0] - pred[:, 1])) + \
                   torch.mean(torch.relu(pred[:, 1] - pred[:, 2])) + \
                   torch.mean(torch.relu(2 - pred))  # bounding the predictions

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss
            loss_accum += loss.detach().cpu().item()

        # Return average loss for epoch
        return loss_accum / (step + 1)

    # Test the trained GNN model on the test dataset
    def te_gnn(self, model, device, loader_atom_bond, loader_bond_angle, test_exp_loader):
        model.eval()
        y_pred, y_true, y_pred_10, y_pred_90 = [], [], [], []
        with torch.no_grad():
            for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle, test_exp_loader)):
                batch_atom_bond = batch[0]
                batch_bond_angle = batch[1]
                test_exp = batch[2]
                batch_atom_bond = batch_atom_bond.to(device)
                batch_bond_angle = batch_bond_angle.to(device)
                test_exp = test_exp.to(device)

                # Forward pass
                pred = model(batch_atom_bond, batch_bond_angle, test_exp)

                # Collect predictions
                y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1,))
                y_pred.append(pred[:, 1].detach().cpu())  # median prediction
                y_pred_10.append(pred[:, 0].detach().cpu())  # lower quantile
                y_pred_90.append(pred[:, 2].detach().cpu())  # upper quantile

            # Concatenate batch results
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            y_pred_10 = torch.cat(y_pred_10, dim=0)
            y_pred_90 = torch.cat(y_pred_90, dim=0)

        # Compute R² and MAE
        R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_pred.mean()) ** 2).sum())
        test_mae = torch.mean((y_true - y_pred) ** 2)

        print(R_square)
        return y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90

    # Evaluate the model on validation or custom dataset (returns only MAE)
    def eval(self, model, device, loader_atom_bond, loader_bond_angle, eval_exp_loader):
        model.eval()
        y_true, y_pred, y_pred_10, y_pred_90 = [], [], [], []
        with torch.no_grad():
            for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle, eval_exp_loader)):
                batch_atom_bond = batch[0]
                batch_bond_angle = batch[1]
                eval_exp = batch[2]
                batch_atom_bond = batch_atom_bond.to(device)
                batch_bond_angle = batch_bond_angle.to(device)
                eval_exp = eval_exp.to(device)

                # Forward pass
                pred = model(batch_atom_bond, batch_bond_angle, eval_exp)

                # Collect predictions
                y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
                y_pred.append(pred[:, 1].detach().cpu())  # median prediction
                y_pred_10.append(pred[:, 0].detach().cpu())  # lower bound
                y_pred_90.append(pred[:, 2].detach().cpu())  # upper bound

        # Concatenate batch results
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred_10 = torch.cat(y_pred_10, dim=0)
        y_pred_90 = torch.cat(y_pred_90, dim=0)

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        # Return Mean Squared Error as numpy float
        return torch.mean((y_true - y_pred) ** 2).data.numpy()

    # Calculate temperature profile from initial temperature, hold time, heating rate, and final temperature
    def calculate_temperatures(self, row):
        temperatures = [row['Initial Temperature']]
        current_temp = row['Initial Temperature']
        time = 0
        while time < 30:
            # Apply heating rate after holding time is reached
            if time >= row['Initial Temperature Hold Time']:
                current_temp += row['Heating Rate']
            # Clip at final temperature
            temperatures.append(min(current_temp, row['Final Temperature']))
            time += 1
        return temperatures[1:]  # Skip the first temperature (initial, already known)

    # Main training/testing workflow based on mode specified in configuration
    def Mode(self):
        if self.GNN_mode == 'Train':
            # Define graph neural network parameters
            nn_params = {
                'num_tasks': self.num_task,
                'num_layers': self.num_layers,
                'emb_dim': self.emb_dim,
                'drop_ratio': self.drop_ratio,
                'graph_pooling': self.graph_pooling,
                'descriptor_dim': 1827
            }

            # Define transformer sequence module parameters
            transformer_params = {
                'seq_len': 32,
                'd_model': 64,
                'num_layers': 3,
                'dropout': 0.1
            }

            # Load data configuration and generate dataset loaders
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
            train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond, \
            train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle, \
            train_exp_loader, valid_exp_loader, test_exp_loader = data.make_gnn_dataset()

            exp_condition_dim = 30
            num_tasks = self.num_task

            # Select model type: MultiModal or MultiSeq
            if self.Whether_sequence == False:
                model = MultiModalModel(graph_nn_params=nn_params, exp_condition_dim=exp_condition_dim, num_tasks=num_tasks).to(device)
            else:
                model = MultiSeqModel(graph_nn_params=nn_params, transfomer_nn_params=transformer_params, num_tasks=num_tasks).to(device)

            # Initialize optimizer and learning rate scheduler
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=self.weight_decay)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

            # Prepare model save folder
            folder_path = self.Train_model_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Train model over multiple epochs
            lowest_mae = float('inf')
            lowest_mae_filename = None
            for epoch in tqdm(range(self.num_iterations)):
                # Training step
                train_mae = self.train_gnn(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, train_exp_loader)

                # Periodically evaluate on validation set
                if (epoch + 1) % 100 == 0:
                    valid_mae = self.eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle, valid_exp_loader)
                    print(train_mae, valid_mae)
                    current_filename = f'/model_save_{epoch + 1}.pth'
                    torch.save(model.state_dict(), folder_path + current_filename)

                    # Save best-performing model
                    if valid_mae < lowest_mae:
                        lowest_mae = valid_mae
                        lowest_mae_filename = current_filename

                scheduler.step()

            print(f"The .pth file with the lowest validation MAE ({lowest_mae}) is: {lowest_mae_filename}")
            return lowest_mae_filename

        # If mode is 'Test': Load trained model and evaluate performance on test set
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

            # Load model structure
            if self.Whether_sequence == False:
                model = MultiModalModel(graph_nn_params=nn_params, exp_condition_dim=exp_condition_dim, num_tasks=num_tasks).to(device)
            else:
                model = MultiSeqModel(graph_nn_params=nn_params, transfomer_nn_params=transformer_params, num_tasks=num_tasks).to(device)

            # Load saved model weights
            model.load_state_dict(torch.load(self.Train_model_path + self.Test_model_path))

            # Generate test dataset
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
            train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond, \
            train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle, \
            train_exp_loader, valid_exp_loader, test_exp_loader = data.make_gnn_dataset()

            # Perform prediction on test set
            y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90 = self.te_gnn(
                model, device, test_loader_atom_bond, test_loader_bond_angle, test_exp_loader
            )

            # Convert to NumPy for evaluation
            y_pred = y_pred.cpu().data.numpy()
            y_true = y_true.cpu().data.numpy()
            y_pred_10 = y_pred_10.cpu().data.numpy()
            y_pred_90 = y_pred_90.cpu().data.numpy()

            # Compute prediction interval coverage
            in_range = np.sum((y_true >= y_pred_10) & (y_true <= y_pred_90))
            ratio = in_range / len(y_true)

            # Print performance metrics
            print('relative_error', np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)))
            print('MAE', np.mean(np.abs(y_true - y_pred) / y_true))
            print('RMSE', np.sqrt(np.mean((y_true - y_pred) ** 2)))

            mae = np.mean(np.abs(y_true - y_pred) / y_true)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
            print(R_square)

            # Plot predicted vs. observed values
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

            # Save figure
            folder_path = self.Outcome_graph_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if self.gnn_train_ratio == 0.9:
                plt.title('GNN_18_1_1')
                plt.savefig(folder_path + "/GNN_18_1_1.svg")
            if self.gnn_train_ratio == 0.8:
                plt.title('Multimodal model')
                plt.savefig(folder_path + "/GNN_8_1_1.svg")

            # Only show the figure if not generating Figure 2c data
            if self.Fig2c_Similarity == False and self.Fig2c_number == False and self.Fig2c_noise == False:
                plt.show()

            return R_square, mae, rmse


        if self.GNN_mode == 'Pre':
            # Set up GNN model hyperparameters
            nn_params = {
                'num_tasks': self.num_task,
                'num_layers': self.num_layers,
                'emb_dim': self.emb_dim,
                'drop_ratio': self.drop_ratio,
                'graph_pooling': self.graph_pooling,
                'descriptor_dim': 1827
            }

            # Set up transformer module hyperparameters
            transformer_params = {
                'seq_len': 32,
                'd_model': 64,
                'num_layers': 3,
                'dropout': 0.1
            }

            exp_condition_dim = 30
            num_tasks = self.num_task

            # Choose model type based on whether sequence information is used
            if self.Whether_sequence == False:
                model = MultiModalModel(
                    graph_nn_params=nn_params,
                    exp_condition_dim=exp_condition_dim,
                    num_tasks=num_tasks
                ).to(device)
            else:
                model = MultiSeqModel(
                    graph_nn_params=nn_params,
                    transfomer_nn_params=transformer_params,
                    num_tasks=num_tasks
                ).to(device)

            # Load pretrained model parameters
            model.load_state_dict(torch.load(self.Train_model_path + self.predict_model_path))

            # Load prediction dataset
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
            data.unknown_descriptor_path = self.unknown_descriptor_path
            data.GNN_mode = self.GNN_mode

            # Construct prediction data loaders
            pre_loader_atom_bond, pre_loader_bond_angle, pre_exp_loader = data.make_gnn_prediction_dataset(
                self.unknown_descriptor_path
            )

            # Run model prediction
            y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90 = self.te_gnn(
                model, device, pre_loader_atom_bond, pre_loader_bond_angle, pre_exp_loader
            )

            # Convert predictions and targets to NumPy arrays
            y_pred = y_pred.cpu().data.numpy()
            y_true = y_true.cpu().data.numpy()
            y_pred_10 = y_pred_10.cpu().data.numpy()
            y_pred_90 = y_pred_90.cpu().data.numpy()

            # Calculate proportion of samples with true value inside prediction interval
            in_range = np.sum((y_true >= y_pred_10) & (y_true <= y_pred_90))
            ratio = in_range / len(y_true)

            # Compute and print evaluation metrics
            print('relative_error', np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)))
            print('MAE', np.mean(np.abs(y_true - y_pred) / y_true))
            print('RMSE', np.sqrt(np.mean((y_true - y_pred) ** 2)))
            R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
            print(R_square)

            # Plot predicted vs observed values
            plt.figure(1, figsize=(3.5, 3.5), dpi=300)
            plt.style.use('ggplot')

            # Use two colors to highlight the last 40 data points
            plt.scatter(y_true[:-40], y_pred[:-40], c='#E79397', s=15, alpha=0.7)
            plt.scatter(y_true[-40:], y_pred[-40:], c='#A797DA', s=15, alpha=0.7)

            # Diagonal reference line (perfect prediction)
            plt.plot(np.arange(0, 22), np.arange(0, 22), linewidth=.5, linestyle='--', color='black')

            # Axis formatting
            plt.yticks(np.arange(0, 22, 2), np.arange(0, 22, 2), fontproperties='Arial', size=8)
            plt.xticks(np.arange(0, 22, 2), np.arange(0, 22, 2), fontproperties='Arial', size=8)
            plt.xlabel('Observed data', fontproperties='Arial', size=8)
            plt.ylabel('Predicted data', fontproperties='Arial', size=8)

            # Annotate with R² and interval coverage ratio
            plt.text(4, 16, "R2 = {:.5f}".format(R_square), fontsize=10, ha='center')
            plt.text(4, 13, "ratio = {:.4f}".format(ratio), fontsize=10, ha='center')

            # Save plot to output folder
            folder_path = self.Outcome_graph_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Customize output path depending on known/unknown input
            if self.unknown_descriptor_path == '../Data/Waiting_pre-known.xlsx':
                plt.title('Multimodal model-Known')
                plt.savefig(folder_path + "/GNN-KnownPrediction.svg")
            else:
                plt.title('Multimodal model-UNKnown')
                plt.savefig(folder_path + "/GNN-unKnownPrediction.svg")

            # Output information
            print(folder_path)
            print('The prediction outcome is {}'.format(y_pred))

            # plt.show()  # Commented out to avoid blocking script execution

# A simple feedforward neural network with residual connections
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)  # Output layer for regression

        self.dropout = nn.Dropout(0.2)  # Dropout regularization
        self.activation = nn.LeakyReLU()  # Activation function

        self.residual = nn.Linear(input_dim, 256)  # Residual connection

    def forward(self, x):
        identity = self.residual(x)  # Save input for residual connection

        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x += identity  # Add residual
        x = self.fc3(x)

        return x


# A wrapper class for training/testing different machine learning models
class Model_ML():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.counter = 0

        # Configuration from args
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

    # Ensure folder for saving models or images exists
    def ensure_directory_exists(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"The directory {directory_path} has been created.")
        else:
            print(f"The directory {directory_path} already exists.")

    # Custom R² function for reporting progress
    def custom_r2(self, y_true, y_pred):
        self.counter += 1
        print('Finished {0}/{1} iterations'.format(self.counter, self.total_iterations))
        return r2_score(y_true, y_pred)

    # Train selected model on given data
    def train(self, x_train, y_train):
        y_cla = y_train[self.ML_label].values.ravel()

        # Train Random Forest
        if self.use_model == 'RF':
            clf = RandomForestRegressor(n_estimators=300, random_state=self.ML_seeds)
            clf.fit(x_train, y_cla)
            y_pre = clf.predict(x_train)
            r2 = r2_score(y_cla, y_pre)
            print('The model is {}, the regression mode'.format(self.use_model))
            print('The Train R2 score is {}'.format(r2))
            self.ensure_directory_exists(self.save_model_file)
            dump(clf, self.save_model_file + 'RF_best_model_regression.joblib')
            return clf

        # Train XGBoost
        if self.use_model == 'XGB':
            clf = XGBRegressor(
                n_estimators=self.XGB_estimators,
                eta=self.XGB_eta_Regress,
                random_state=self.ML_seeds,
                max_depth=self.XGB_depth_Regress,
                subsample=self.XGB_subsample_Regress
            )
            clf.fit(x_train, y_cla)
            y_pre = clf.predict(x_train)
            r2 = r2_score(y_cla, y_pre)
            print('The model is {}, the regression mode'.format(self.use_model))
            print('The Train R2 score is {}'.format(r2))
            self.ensure_directory_exists(self.save_model_file)
            dump(clf, self.save_model_file + 'XGB_best_model_regression.joblib')
            return clf

        # Train LightGBM
        if self.use_model == 'LGB':
            clf = LGBMRegressor(
                n_estimators=self.LGB_estimators,
                random_state=self.ML_seeds,
                learning_rate=self.LGB_learning_rate_Regress,
                max_depth=self.LGB_max_depth_Regress,
                min_child_samples=self.LGB_min_child_samples_Regress,
                num_leaves=self.LGB_num_leaves_Regress
            )
            clf.fit(x_train, y_cla)
            y_pre = clf.predict(x_train)
            r2 = r2_score(y_cla, y_pre)
            print('The model is {}, the regression mode'.format(self.use_model))
            print('The Train R2 score is {}'.format(r2))
            self.ensure_directory_exists(self.save_model_file)
            dump(clf, self.save_model_file + 'LGB_best_model_regression.joblib')
            return clf

        # Train ANN
        if self.use_model == 'ANN':
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

            # Convert to PyTorch tensors
            x_train = torch.tensor(x_train.values, dtype=torch.float32).to(self.device)
            y_train_cla = torch.tensor(y_train[self.ML_label].values.ravel(), dtype=torch.float32).to(self.device)
            train_dataset = TensorDataset(x_train, y_train_cla)
            train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

            x_valid = torch.tensor(x_valid.values, dtype=torch.float32).to(self.device)
            y_valid = torch.tensor(y_valid[self.ML_label].values.ravel(), dtype=torch.float32).to(self.device)
            valid_dataset = TensorDataset(x_valid, y_valid)
            valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, drop_last=True)

            model = SimpleNN(x_train.shape[1]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False)

            best_val_loss = float('inf')
            best_model_state = copy.deepcopy(model.state_dict())

            # Training loop
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
                    print(f'Epoch {epoch}/{self.max_iteration} - Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

            model.load_state_dict(best_model_state)
            self.ensure_directory_exists(self.save_model_file)
            torch.save(model.state_dict(), self.save_model_file + 'ANN_best_model_regression.pth')


    # Test the trained model on test data
    def test(self, x_test, y_test):
        y_cla = y_test[self.ML_label].values.ravel()

        if self.use_model == 'ANN':
            model = SimpleNN(x_test.shape[1]).to(self.device)
            model.load_state_dict(torch.load(self.save_model_file + 'ANN_best_model_regression.pth'))
            model.eval()
            summary(model, input_size=(x_test.shape[1],))  # Print model structure
            x = torch.tensor(x_test.values, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                y_pre = model(x).detach().cpu()
        else:
            # Load classical ML model
            model = load(self.save_model_file + '{}_best_model_regression.joblib'.format(self.use_model))
            y_pre = model.predict(x_test)

        # Plot predicted vs. observed results
        plt.figure(1, figsize=(3.5, 3.5), dpi=300)
        plt.style.use('ggplot')
        r2 = r2_score(y_cla, y_pre)

        # Filter predictions to only plot values within a certain range
        filtered_list1, filtered_list2 = zip(
            *[(item1, item2) for index, (item1, item2) in enumerate(zip(y_cla, y_pre)) if item1 <= 22]
        )
        plt.scatter(filtered_list1, filtered_list2, c='#8983BF', s=15, alpha=0.4)
        plt.plot(np.arange(0, 22), np.arange(0, 22), linewidth=.5, linestyle='--', color='black')
        plt.yticks(np.arange(0, 22, 2), np.arange(0, 22, 2), fontproperties='Arial', size=8)
        plt.xticks(np.arange(0, 22, 2), np.arange(0, 22, 2), fontproperties='Arial', size=8)
        plt.xlabel('Peak_time-Observed', fontproperties='Arial', size=8)
        plt.ylabel('Peak_time-Predicted', fontproperties='Arial', size=8)
        plt.text(4, 16, "R2 = {:.5f}".format(r2), fontsize=10, ha='center')
        plt.title('{}'.format(self.use_model))

        print('The model is {}, the regression mode'.format(self.use_model))
        print('The test R2 score is {}'.format(r2))

        # Save prediction image
        self.ensure_directory_exists(self.image_output)
        plt.savefig(self.image_output + "{}_231023.svg".format(self.use_model))
        plt.show()

        return r2, y_pre