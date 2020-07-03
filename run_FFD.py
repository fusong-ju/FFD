#!/usr/bin/env python
import sys
import numpy as np
import click
from task.prediction import Prediction
from task.task import Task


@click.command()
@click.option("-f",
              "--fasta_path",
              required=True,
              type=click.Path(exists=True),
              help="Fasta format sequence")
@click.option("-g",
              "--geo_path",
              required=True,
              type=click.Path(exists=True),
              help="CopulaNet prediction")
@click.option("-o",
              "--outdir",
              required=True,
              type=click.Path(),
              help="Output directory")
@click.option("-n",
              "--n_structs",
              default=1,
              type=int,
              help="Number of decoys to generate")
@click.option("-p", "--pool_size", default=-1, type=int)
@click.option("-d",
              "--device",
              default=0,
              type=int,
              help="GPU device to use (-1 for CPU)")
@click.option("--snapshot", type=int, is_flag=True)
def main(fasta_path, geo_path, outdir, n_structs, pool_size, device, snapshot):
    geo_npz = np.load(geo_path)
    if device == -1:
        device = "cpu"
    pred = Prediction(geo_npz)
    if pool_size == -1:
        pool_size = n_structs
    assert not snapshot or n_structs == 1, "n_structs must be set to 1 if snapshot"
    task = Task(fasta_path=fasta_path,
                prediction=pred,
                outdir=outdir,
                n_structs=n_structs,
                pool_size=pool_size,
                device=device,
                is_snapshot=snapshot)
    task.start()


if __name__ == "__main__":
    main()
