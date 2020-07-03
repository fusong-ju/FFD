#!/bin/bash
set -e

outdir=output
name=1ctfA

mkdir -p $outdir

../run_FFD.py \
  --fasta_path $name.fasta \
  --geo_path $name.npz \
  --outdir $outdir \
  --n_structs 10 \
  --device 0
