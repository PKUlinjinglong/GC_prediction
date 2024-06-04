
# 多模态模型 README

这是一个多模态模型，用于预测分子在 30 m × 0.25 mm WondaCAP-5 毛细管柱（内部涂有 0.25 µm 的 5% 苯基-95% 二甲基聚硅氧烷固定相）下，在各种升温曲线下的保留时间。

## 环境设置

需要满足的环境包括：
- **PyTorch 版本**：2.3.0+cu121
- **torch_geometric 版本**：2.5.3

## 库的安装方法

使用以下命令来安装所需的库：

```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip install torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## 代码运行方法

1. 在 `./Model/config.py` 文件里面设置配置参数。

2. 在 `main.py` 中运行主函数：

3. 修改配置文件：

在 `./Model/config.py` 文件中设置配置参数，确保所有参数都已正确配置。

4. 运行代码：

在项目的根目录下运行主函数：

```bash
python main.py
```

## 使用说明

1. 在 `predict_mole_smi` 列表中输入你所感兴趣的化学分子的 SMILES 结构：

    ```python
    predict_mole_smi = ['CCCCCCC', 'CCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCC']
    ```

2. 在 `exp_cond` 列表里输入升温曲线的条件（初始温度、末态温度、升温速率、初始温度持续时间）：

    ```python
    exp_cond = [40, 200, 20, 2]
    ```

3. 运行主函数即可预测出该化学分子在特定升温曲线下的保留时间：

    ```bash
    python main.py
    ```

4. 修改 `GNN.GNN_mode` 参数即可进行训练和测试：

    - **Train 模式**：

        ```python
        GNN.GNN_mode = 'Train'
        data.prepare_gnn_3D()
        GNN.num_iterations = 1500
        GNN.Mode()
        ```

    - **Test 模式**：

        ```python
        GNN.GNN_mode = 'Test'
        GNN.Test_model_path = '/model_save_600.pth'
        GNN.Mode()
        ```

    - **Pre 模式**：

        ```python
        GNN.GNN_mode = 'Pre'
        GNN.predict_model_path = '/model_save_600.pth'
        predict_mole_smi = ['CCCCCCC', 'CCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCC']
        exp_cond = [40, 200, 20, 2]
        generate_excel(predict_mole_smi, exp_cond)
        GNN.unknown_descriptor_path = './Data/waiting_pre.xlsx'
        GNN.Mode()
        ```

## 注意事项

- 在运行代码之前，请确保在 `./Model/config.py` 文件中正确设置所有配置参数。
- 确保安装的 PyTorch 和 torch_geometric 版本符合要求，否则可能会导致兼容性问题。

通过这些步骤，你可以使用此多模态模型预测化学分子在特定升温曲线下的保留时间。
