from typing import List
from openmm import app

AA = [
    "ala",
    "arg",
    "asn",
    "asp",
    "cys",
    "gln",
    "glu",
    "gly",
    "his",
    "ile",
    "leu",
    "lys",
    "met",
    "phe",
    "pro",
    "pyl",
    "ser",
    "sec",
    "thr",
    "trp",
    "tyr",
    "val",
    "asx",
    "glx",
    "xaa",
    "xle",
]

BB = ["[CA:", "[C:", "[N:"]


def get_residue_indices(
    atom_list: app.internal.charmm.topologyobjects.AtomList,
    name: List[str] or str = AA,
    include_hydrogens: bool = True,
) -> List[int]:
    """
    Returns a list of indices that correspond to a given residue in the pdb file. only give atom list if you want all the protein residues.

    Args:
        atom_list (app.internal.charmm.topologyobjects.AtomList): atom list that should be searched.
        name (List[str]orstr, optional): residue name. if none is given, the atom list is searched for amino acid residues. Defaults to aa_list.

    Returns:
        List[int]: list of indices in the pdb file.
    """
    ret = []
    for i, atom in enumerate(atom_list):
        if not include_hydrogens and "[H" in str(atom):
            continue
        if isinstance(name, str):
            if name.lower() in str(atom).lower():
                ret.append(i)
        else:
            if any(aa.lower() in str(atom).lower() for aa in name):
                ret.append(i)
    return ret


def get_backbone_indices(
    atom_list: app.internal.charmm.topologyobjects.AtomList,
) -> List[int]:
    """
    Returns list of protein backbone indices. will also return every other

    Args:
        atom_list (app.internal.charmm.topologyobjects.AtomList): [description]

    Returns:
        List[int]: [description]
    """
    ret = []
    prot_res_idx = get_residue_indices(atom_list=atom_list)
    for it, atom in enumerate(prot_res_idx):
        if any(bb in str(atom) for bb in BB):
            ret.append(it)
    return ret
