from typing import List
import openmm.app as app

aa_list = [
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


def get_residue_indices(
    atom_list: app.internal.charmm.topologyobjects.AtomList,
    name: List[str] or str = aa_list,
    include_hydrogens: bool = True,
) -> List[int]:
    """
    Returns a list of indices that correspond to a given residue in the pdb file.

    Args:
        atom_list (app.internal.charmm.topologyobjects.AtomList): atom list that should be searched.
        name (List[str]orstr, optional): residue name. if none is given, the atom list is searched for amino acid residues. Defaults to aa_list.

    Returns:
        List[int]: list of indices in the pdb file.
    """
    ret = []
    for i, atom in enumerate(atom_list):
        if isinstance(name, str):
            if not include_hydrogens and "[H" in str(atom):
                continue
            if name.lower() in str(atom).lower():
                ret.append(i)
        else:
            if not include_hydrogens and "[H" in str(atom):
                continue
            if any(aa.lower() in str(atom).lower() for aa in name):
                ret.append(i)
    return ret
