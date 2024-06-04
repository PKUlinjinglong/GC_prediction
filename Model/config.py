import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # --------------parameters for path-----------------------
    parser.add_argument('--clean_data_path', type=str, default='../Data/Clean_database.csv')
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
    parser.add_argument('--Train_model_path', type=str, default='../Outcome/GNN_seq240408_250c')
    parser.add_argument('--Test_model_path', type=str, default='/model_save_600.pth')
    parser.add_argument('--predict_model_path', type=str, default='/model_save_600.pth')
    parser.add_argument('--recommend_model_path', type=str, default='/model_save_600.pth')
    parser.add_argument('--Re_smiles_A', type=str, default='O=CC=CC1=CC=CC=C1')
    parser.add_argument('--Re_smiles_B', type=str, default='O=C/C=C/C1=CC=CC=C1')
    parser.add_argument('--Outcome_graph_path', type=str, default='../Outcome/GNN_seq240408_250c/Test_Graph')

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