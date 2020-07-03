import os

import numpy as np
import Bio.PDB
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue


def read_fasta(fasta_path):
    lines = open(fasta_path).readlines()
    assert len(lines) == 2, "Only single line sequence is supported"
    return lines[1].strip()


def write_pdb(seq, N, CA, C, CB, path, info=None):
    chain = Chain("A")
    for i in range(len(seq)):
        resname = Bio.PDB.Polypeptide.one_to_three(seq[i])
        residue = Residue((" ", i + 1, " "), resname, "    ")
        residue.add(Atom("N", N[i], 0.0, 1.0, " ", " N", 0, "N"))
        residue.add(Atom("CA", CA[i], 0.0, 1.0, " ", " CA", 0, "C"))
        if seq[i] != "G":
            residue.add(Atom("CB", CB[i], 0.0, 1.0, " ", " CB", 0, "C"))
        residue.add(Atom("C", C[i], 0.0, 1.0, " ", " C", 0, "C"))
        chain.add(residue)
    model = Model(0)
    model.add(chain)
    structure = Structure("X")
    structure.add(model)
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(path)
    if info:
        with open(path, "a") as fp:
            fp.write(f"# {info}\n")
