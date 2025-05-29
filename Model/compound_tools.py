import os
from collections import OrderedDict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem.Descriptors import rdMolDescriptors
import os
from mordred import Calculator, descriptors,is_missing
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.cluster import KMeans
import tqdm
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import numpy as np
from mordred import Calculator, descriptors,is_missing

calc = Calculator(descriptors, ignore_3D=False)
DAY_LIGHT_FG_SMARTS_LIST = [
        # C
        "[CX4]",
        "[$([CX2](=C)=C)]",
        "[$([CX3]=[CX3])]",
        "[$([CX2]#C)]",
        # C & O
        "[CX3]=[OX1]",
        "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "[CX3](=[OX1])C",
        "[OX1]=CN",
        "[CX3](=[OX1])O",
        "[CX3](=[OX1])[F,Cl,Br,I]",
        "[CX3H1](=O)[#6]",
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "[NX3][CX3](=[OX1])[#6]",
        "[NX3][CX3]=[NX3+]",
        "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "[NX3][CX3](=[OX1])[OX2H0]",
        "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "[CX3](=O)[O-]",
        "[CX3](=[OX1])(O)O",
        "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
        "C[OX2][CX3](=[OX1])[OX2]C",
        "[CX3](=O)[OX2H1]",
        "[CX3](=O)[OX1H0-,OX2H1]",
        "[NX3][CX2]#[NX1]",
        "[#6][CX3](=O)[OX2H0][#6]",
        "[#6][CX3](=O)[#6]",
        "[OD2]([#6])[#6]",
        # H
        "[H]",
        "[!#1]",
        "[H+]",
        "[+H]",
        "[!H]",
        # N
        "[NX3;H2,H1;!$(NC=O)]",
        "[NX3][CX3]=[CX3]",
        "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "[NX3][$(C=C),$(cc)]",
        "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
        "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
        "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
        "[CH2X4][CX3](=[OX1])[NX3H2]",
        "[CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[CH2X4][SX2H,SX1H0-]",
        "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
        "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
        "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "[CH2X4][CH2X4][SX2][CH3X4]",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH2X4][OX2H]",
        "[NX3][CX3]=[SX1]",
        "[CHX4]([CH3X4])[OX2H]",
        "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
        "[CHX4]([CH3X4])[CH3X4]",
        "N[CX4H2][CX3](=[OX1])[O,N]",
        "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
        "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
        "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
        "[#7]",
        "[NX2]=N",
        "[NX2]=[NX2]",
        "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "[NX3][NX3]",
        "[NX3][NX2]=[*]",
        "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "[NX3+]=[CX3]",
        "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
        "[NX1]#[CX2]",
        "[CX1-]#[NX2+]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[NX2]=[OX1]",
        "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        # O
        "[OX2H]",
        "[#6][OX2H]",
        "[OX2H][CX3]=[OX1]",
        "[OX2H]P",
        "[OX2H][#6X3]=[#6]",
        "[OX2H][cX3]:[c]",
        "[OX2H][$(C=C),$(cc)]",
        "[$([OH]-*=[!#6])]",
        "[OX2,OX1-][OX2,OX1-]",
        # P
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        # S
        "[S-][CX3](=S)[#6]",
        "[#6X3](=[SX1])([!N])[!N]",
        "[SX2]",
        "[#16X2H]",
        "[#16!H0]",
        "[#16X2H0]",
        "[#16X2H0][!#16]",
        "[#16X2H0][#16X2H0]",
        "[#16X2H0][!#16].[#16X2H0][!#16]",
        "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "[SX4](C)(C)(=O)=N",
        "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "[#16X2][OX2H,OX1H0-]",
        "[#16X2][OX2H0]",
        # X
        "[#6][F,Cl,Br,I]",
        "[F,Cl,Br,I]",
        "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
    ]


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Gasteiger Partial Charges, also known as Gasteiger-Marsili partial charges, are a method used to estimate 
    the partial charge on each atom in a molecule. They are based on an empirical approach to approximate the 
    charge distribution on atoms.

    Applications:

    - Molecular modeling and drug design:
    Understanding the charge distribution within a molecule helps elucidate its structure, stability, and reactivity,
    which is crucial for the design and discovery of new drugs.

    - Predicting molecular interactions:
    Partial charges can be used to predict non-covalent interactions between molecules, such as hydrogen bonding 
    and van der Waals forces. This is important for understanding protein–ligand and protein–protein interactions.

    - QSAR/QSPR studies:
    In Quantitative Structure–Activity Relationship (QSAR) and Quantitative Structure–Property Relationship (QSPR) 
    research, partial charges can serve as descriptors to predict biological activity or other properties of molecules.

    - Reaction mechanism studies:
    When investigating chemical reaction mechanisms, understanding the charge distribution within a molecule 
    helps in analyzing reaction pathways and the characteristics of transition states.

    - Physicochemical property prediction:
    Partial charges can also be used to predict and interpret physicochemical properties of substances, such as 
    solubility, melting point, and boiling point.

    Code implementation:

    The Gasteiger algorithm is used to calculate the partial charges for each atom in a given RDKit molecule object.

    - nIter: The number of iterations for the Gasteiger algorithm (default is 12).
    - throwOnParamFailure: A boolean indicating whether to raise an exception if parameter initialization fails.

    The first line is responsible for computing the partial charges for each atom.  
    The second line is responsible for accessing the partial charge of each atom.  
    Without the first line, the `.mol` object does not contain partial charge information and thus cannot be accessed.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """
    Args:
        smiles: smiles sequence.
    Returns:
        inchi.
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry 去除立体化学信息
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None:  # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

def check_smiles_validity(smiles):
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively.
    Args:
        mol: rdkit mol object.
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one.
    Args:
        mol_list(list): a list of rdkit mol object.
    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return alist.index(elem)    #寻找列表当中某元素的索引
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    """
    CompoundKit: A utility class for extracting atom and bond features,
    fingerprints, and other molecular descriptors using RDKit.
    """

    # Atom-level categorical feature vocabularies
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": list(range(0, 11)) + ['misc'],
        "explicit_valence": list(range(0, 13)) + ['misc'],
        "formal_charge": list(range(-5, 11)) + ['misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": list(range(0, 13)) + ['misc'],
        "is_aromatic": [0, 1],
        "total_numHs": list(range(0, 9)) + ['misc'],
        'num_radical_e': list(range(0, 5)) + ['misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': list(range(0, 9)) + ['misc'],
        'in_num_ring_with_size3': list(range(0, 9)) + ['misc'],
        'in_num_ring_with_size4': list(range(0, 9)) + ['misc'],
        'in_num_ring_with_size5': list(range(0, 9)) + ['misc'],
        'in_num_ring_with_size6': list(range(0, 9)) + ['misc'],
        'in_num_ring_with_size7': list(range(0, 9)) + ['misc'],
        'in_num_ring_with_size8': list(range(0, 9)) + ['misc'],
    }

    # Bond-level categorical feature vocabularies
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }

    # Atom-level continuous features (float descriptors)
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']

    # SMARTS-based functional group patterns
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    # Default fingerprint bit lengths
    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    # Periodic table object from RDKit
    period_table = Chem.GetPeriodicTable()

    @staticmethod
    def get_atom_value(atom, name):
        """Return the specified property value from an RDKit atom object."""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """Return the feature index (ID) for a given atom property."""
        assert name in CompoundKit.atom_vocab_dict, f"{name} not found in atom_vocab_dict"
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """Return the size (vocabulary length) of a given atom feature."""
        assert name in CompoundKit.atom_vocab_dict, f"{name} not found in atom_vocab_dict"
        return len(CompoundKit.atom_vocab_dict[name])

    # -------------------- Bond Feature Extraction -------------------- #
    
    @staticmethod
    def get_bond_value(bond, name):
        """Return the specified property value from an RDKit bond object."""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """Return the feature index (ID) for a given bond property."""
        assert name in CompoundKit.bond_vocab_dict, f"{name} not found in bond_vocab_dict"
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """Return the size (vocabulary length) of a given bond feature."""
        assert name in CompoundKit.bond_vocab_dict, f"{name} not found in bond_vocab_dict"
        return len(CompoundKit.bond_vocab_dict[name])

    # -------------------- Fingerprint Computation -------------------- #

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """Return a binary Morgan fingerprint (default length: 200 bits)."""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """Return a binary Morgan fingerprint with 2048 bits."""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups
    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """
        Count Daylight-defined functional groups in the molecule.

        This method computes the number of occurrences of each functional group 
        defined in CompoundKit.day_light_fg_mo_list for the given molecule.

        It uses Chem.Mol.GetSubstructMatches to find all matches and returns a list of counts.
        """
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """
        Return a list of shape (N_atoms, 6).

        For each atom, returns a list of 6 values indicating how many times
        the atom appears in rings of size 3 to 8 respectively.

        Example:
            result[0] = [1, 0, 0, 2, 0, 0]
            → atom 0 is in one 3-membered ring and two 6-membered rings.
        """
        rings = mol.GetRingInfo()
        rings_info = [r for r in rings.AtomRings()]
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):  # Check rings of size 3 to 8
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9  # Cap to 'misc' threshold
                atom_result.append(num_of_ring_at_ringsize)
            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """
        Convert an RDKit atom object to a dictionary of feature values.

        Each entry corresponds to a predefined atomic feature in CompoundKit.atom_vocab_dict,
        with the value being either a categorical index or a float descriptor.
        """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(
                CompoundKit.atom_vocab_dict['valence_out_shell'],
                CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
            ),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """
        Get a list of atom feature dictionaries for each atom in the molecule.

        This includes basic atomic descriptors as well as ring membership counts (3–8-membered).
        Also invokes Gasteiger charge computation before accessing charge-related features.
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for atom in mol.GetAtoms():
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        # Append ring size features to each atom's dictionary
        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])
        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """
        Safely retrieve Gasteiger partial charge from atom.

        Handles NaN and infinite values by replacing them with safe fallbacks.
        """
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:  # NaN check
            pc = 0
        if pc == float('inf'):
            pc = 10
        return pc

class Compound3DKit(object):
    """3D utility kit for handling molecular conformations and spatial features."""

    @staticmethod
    def get_atom_poses(mol, conf):
        """
        Extract 3D coordinates of atoms from a conformer object.
        If dummy atom (atomic number = 0) is found, return zero vector for all atoms.
        """
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """
        Return MMFF atom positions.
        NOTE: The molecule may be modified during this process.
        """
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """Compute and return 2D atom coordinates."""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """Calculate bond lengths between atom pairs given their 3D coordinates."""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        return np.array(bond_lengths, dtype='float32')

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """
        Calculate angles between consecutive bonds (superedges) in the graph.
        dir_type:
            'HT': tail-head (e.g., i→j then j→k)
            'HH': head-head (e.g., i→j then k→j)
        Returns:
            super_edges: list of edge pair indices
            bond_angles: angle between the edge vectors
            bond_angle_dirs: True if direction is tail→head→head
        """

        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # avoid division by zero
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges, bond_angles, bond_angle_dirs = [], [], []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)

            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])

        if len(super_edges) == 0:
            return np.zeros([0, 2], 'int64'), np.zeros([0, ], 'float32'), []
        else:
            return np.array(super_edges, 'int64'), np.array(bond_angles, 'float32'), bond_angle_dirs


def new_smiles_to_graph_data(smiles, **kwargs):
    """
    Convert a SMILES string to graph data dictionary.
    Returns None if conversion fails.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return new_mol_to_graph_data(mol)


def new_mol_to_graph_data(mol):
    """
    Convert an RDKit Mol object to graph data.
    
    Returns a dictionary containing:
        - atom features
        - bond features
        - edges (adjacency list)
        - molecular fingerprints
        - functional group counts
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())
    data = {name: [] for name in atom_id_names + bond_id_names}
    data['edges'] = []

    # Atom features
    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    # Bond features and edge list
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        data['edges'] += [(i, j), (j, i)]  # undirected: add both directions
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    # Add self-loops
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'].append((i, i))
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dims([name])[0] - 1  # use last index for self-loop
        data[name] += [bond_feature_id] * N

    # Convert to NumPy arrays
    for name in CompoundKit.atom_vocab_dict.keys():
        data[name] = np.array(data[name], dtype='int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], dtype='float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], dtype='int64')
    data['edges'] = np.array(data['edges'], dtype='int64')

    # Molecular-level features
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), dtype='int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), dtype='int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), dtype='int64')

    return data

def mol_to_graph_data(mol):
    """
    Convert RDKit Mol object to graph data with atom/bond features and fingerprints.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    # Feature names
    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence",
        "formal_charge", "hybridization", "implicit_valence",
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = ["bond_dir", "bond_type", "is_in_ring"]

    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    # Atom features
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # +1 for OOV protection
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)   # Normalize mass

    # Bond features (add both directions)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1
            data[name] += [bond_feature_id] * 2

    # Self-loops
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2  # self-loop index
        data[name] += [bond_feature_id] * N

    # Handle molecule with no bonds
    if len(data['edges']) == 0:
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    # Convert to numpy arrays
    for name in atom_id_names:
        data[name] = np.array(data[name], dtype='int64')
    data['mass'] = np.array(data['mass'], dtype='float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], dtype='int64')
    data['edges'] = np.array(data['edges'], dtype='int64')

    # Molecular fingerprints and functional group counts
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), dtype='int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), dtype='int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), dtype='int64')
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    """
    Convert mol and 3D atomic positions to graph data including bond lengths and bond angles.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)
    data['atom_pos'] = np.array(atom_poses, dtype='float32')

    # Calculate bond lengths from 3D coordinates
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])

    # Calculate bond angle graph
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
        Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'], dir_type=dir_type)
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, dtype='float32')

    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    """
    Generate GeoGNN graph data with 3D conformer using MMFF optimization.
    If molecule has >400 atoms, fallback to 2D coordinates.
    """
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')


def mol_to_geognn_graph_data_raw3d(mol):
    """
    Generate GeoGNN graph data from existing conformer (without re-optimization).
    """
    atom_poses = Compound3DKit.get_atom_poses(mol, mol.GetConformer())
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')


def obtain_3D_mol(smiles, name):
    """
    Generate a 3D conformer from a SMILES string and save it as a .mol file.
    """
    mol = AllChem.MolFromSmiles(smiles)
    new_mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(new_mol)
    AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)
    Chem.MolToMolFile(new_mol, name + '.mol')
    return new_mol


def mord(mol, nBits=1826, errors_as_zeros=True):
    """
    Calculate molecular descriptors using Mordred. Handle NaN and errors.
    """
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)


def save_3D_mol(all_smile, mol_save_dir):
    """
    Attempt to convert a list of SMILES strings into 3D mol files and save them.
    Return a list of indices for molecules that failed to convert.
    """
    index = 0
    bad_conformer = []
    pbar = tqdm(all_smile)
    os.makedirs(mol_save_dir, exist_ok=True)
    for smiles in pbar:
        try:
            obtain_3D_mol(smiles, f'{mol_save_dir}/3D_mol_{index}')
        except ValueError:
            bad_conformer.append(index)
            index += 1
            continue
        index += 1
    return bad_conformer  # return indices of failed conformers


def save_dataset(charity_smile, mol_save_dir, charity_name, moder_name, bad_conformer):
    """
    Save both GeoGNN graph data and Mordred descriptors for a list of molecules.
    Molecules that failed 3D generation (in bad_conformer) are skipped.
    """
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
        index += 1

    dataset_mord = np.array(dataset_mord)
    np.save(f"{charity_name}.npy", dataset, allow_pickle=True)  # Save graph data
    np.save(f"{moder_name}.npy", dataset_mord)                  # Save molecular descriptors


if __name__ == "__main__":
    # Example usage
    smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
    mol = AllChem.MolFromSmiles(smiles)
    data = mol_to_geognn_graph_data_MMFF3d(mol)
    # for key, value in data.items():
    #     print(key, value.shape)
