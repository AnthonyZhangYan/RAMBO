import torch
from rdkit import Chem
from rdkit.Chem import AllChem

_cache = {}

def _prepare_backbone(n):
    if n in _cache:
        return _cache[n]

    smiles = "C" * n
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = False
    AllChem.EmbedMolecule(mol, params)

    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
    _cache[n] = (mol, mp)
    return _cache[n]

def _set_torsions(mol, angles_deg):
    """Set torsion angles on the pre-embedded backbone."""
    conf = mol.GetConformer()
    D = len(angles_deg)

    for i in range(D):
        a1, a2, a3, a4 = i, i+1, i+2, i+3
        AllChem.SetDihedralDeg(conf, a1, a2, a3, a4, float(angles_deg[i]))

def _optimize_except_torsions(mol, mp):
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)

    torsions = []
    n = mol.GetNumAtoms()
    D = n - 3
    conf = mol.GetConformer()
    for i in range(D):
        a1, a2, a3, a4 = i, i+1, i+2, i+3
        target = AllChem.GetDihedralDeg(conf, a1, a2, a3, a4)
        torsions.append((a1, a2, a3, a4, target))

    for (a1,a2,a3,a4,target) in torsions:
        ff.AddTorsionConstraint(a1,a2,a3,a4, True, target, target, 1.0)

    ff.Initialize()
    ff.Minimize(maxIts=200)
    return ff.CalcEnergy()

def torsion_energy(angles, n=15):
    """
    angles: 1D torch tensor of shape (n-3,)
    Return: -MMFF94 energy (float32)  (maximize(-E))
    """
    angles = angles.detach().cpu().numpy()

    mol, mp = _prepare_backbone(n)

    mol = Chem.Mol(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=False)  # deterministic copy coords

    _set_torsions(mol, angles)

    try:
        energy = _optimize_except_torsions(mol, mp)
    except:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        energy = ff.CalcEnergy()

    return torch.tensor(-energy, dtype=torch.float32)
