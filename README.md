
# Multimodal Model README

This is a multimodal model for predicting the retention time of molecules under various temperature ramping curves.

## Environment Setup

Required environment includes:
- **PyTorch Version**: 2.3.0+cu121
- **torch_geometric Version**: 2.5.3

## Library Installation

Use the following commands to install the required libraries:

```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip install torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## Running the Code

1. Set configuration parameters in the `./Model/config.py` file.

2. Run `main.py` in the root directory of the project:

```bash
python main.py
```

## Usage Instructions

1. Input the SMILES strings of the chemical molecules you are interested in into the `predict_mole_smi` list (these can be exported from ChemDraw):

    ```python
    predict_mole_smi = ['CCCCCCC', 'CCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCC']
    ```

2. Input the conditions of the temperature ramping curve into the `exp_cond` list (initial temperature, final temperature, heating rate, initial temperature holding time):

    ```python
    exp_cond = [40, 200, 20, 2]
    ```

3. Run `main.py` to predict the retention time of the chemical molecule under the specified temperature ramping curve:

    ```bash
    python main.py
    ```

4. Modify the `GNN.GNN_mode` parameter to perform training and testing:

    - **Train Mode**:

        ```python
        GNN.GNN_mode = 'Train'
        data.prepare_gnn_3D()
        GNN.num_iterations = 1500
        GNN.Mode()
        ```

    - **Test Mode**:

        ```python
        GNN.GNN_mode = 'Test'
        GNN.Test_model_path = '/model_save_600.pth'
        GNN.Mode()
        ```

    - **Pre Mode**:

        ```python
        GNN.GNN_mode = 'Pre'
        GNN.predict_model_path = '/model_save_600.pth'
        predict_mole_smi = ['CCCCCCC', 'CCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCC']
        exp_cond = [40, 200, 20, 2]
        generate_excel(predict_mole_smi, exp_cond)
        GNN.unknown_descriptor_path = './Data/waiting_pre.xlsx'
        GNN.Mode()
        ```

## Notes

- Before running the code, ensure all configuration parameters are correctly set in the `./Model/config.py` file.
- Ensure that the installed versions of PyTorch and torch_geometric meet the requirements to avoid compatibility issues.

By following these steps, you can use this multimodal model to predict the retention time of chemical molecules under specific temperature ramping curves.
