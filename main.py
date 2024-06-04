from Model.data_process import Dataset_process,generate_excel
from Model.config import parse_args
from Model.model import GNN_process
import warnings
warnings.filterwarnings("ignore")

def main():
    config = parse_args()
    data = Dataset_process(config)
    GNN = GNN_process(config)
    GNN.GNN_mode = 'Pre'         # Optional parameters:  Train/Test/Pre
    """Train"""
    # data.prepare_gnn_3D()
    # GNN.num_iterations = 1500
    # GNN.Mode()
    """Test"""
    # GNN.Test_model_path = '/model_save_600.pth'
    # GNN.Mode()
    """Pre"""
    GNN.predict_model_path = '/model_save_600.pth'
    predict_mole_smi = ['CCCCCCC','CCCCCCCCCCCC','CCCCCCCCCCCCCCCCCCC']
    exp_cond = [40,200,20,2]
    generate_excel(predict_mole_smi, exp_cond)
    GNN.unknown_descriptor_path = './Data/waiting_pre.xlsx'
    GNN.Mode()

if __name__ == "__main__":
    main()