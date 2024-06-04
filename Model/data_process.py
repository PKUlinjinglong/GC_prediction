from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem, PandasTools, Descriptors
import shutil
from compound_tools import *
import os
from mordred import Calculator, descriptors,is_missing
import random
from rdkit import Chem, DataStructs


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


def generate_excel(predict_mole_smi, exp_cond):
    initial_temp = exp_cond[0]
    final_temp = exp_cond[1]
    heating_rate = exp_cond[2]
    retention_time = exp_cond[3]
    temperature_series = list(range(initial_temp, final_temp + 1, heating_rate))
    while len(temperature_series) < 30:
        temperature_series.append(temperature_series[-1])
    temperature_series = temperature_series[:30]
    data = []
    for smi in predict_mole_smi:
        row = [smi] + temperature_series + [retention_time]
        data.append(row)
    df = pd.DataFrame(data)
    output_path = './Data/waiting_pre.xlsx'
    df.to_excel(output_path, index=False, header=False)

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