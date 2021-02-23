"""
This script builds a dataset in pickle
format from a folder with FASTA files. The
desired label of the file must be in the file ID
after the accession number separated by a dot.

:param dataset: Name of the Dataset.
:param data_path: Path of the folder with the sequences.
:returns: None

Example: python build_dp.py --data_path = '../data/Influenza'
"""

import os
from Bio import SeqIO
import pickle
import argparse


def replace(seq):
    """
    This function ignores all the symbols that are not A, C, G, T
    in the sequence
    :param seq: Original sequence
    :return: Sequences with only ACGT characters.
    """
    newseq = []
    accepted_bp = ['A', 'C', 'G', 'T']
    for l in seq:
        if l in accepted_bp:
            newseq.append(l)

    return ''.join(newseq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, default='None')
    args = parser.parse_args()

    seq_dir = args.data_path
    data = []

    labels = os.listdir(seq_dir)

    for label in labels:
        for file in os.listdir(os.path.join(seq_dir, label)):
            filename = os.path.join(seq_dir, label, file)
            print(filename)

            # Read FASTA file.
            fasta = SeqIO.parse(filename, "fasta")

            for sequence in fasta:
                accession_number = sequence.id.split('.')[0]

                # Get Sequence
                seq = str(sequence.seq)
                seq = seq.upper()
                seq = replace(seq)

                data.append((label, seq, accession_number))

    pathTrain = os.path.join(seq_dir, 'train.p')
    pickle.dump(data, open(pathTrain, "wb"))


if __name__ == '__main__':
    main()
