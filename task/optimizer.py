import os
import math
import functools
import logging

import torch

from .scheduler import cosine_schedule
from utils.io_tool import write_pdb


def closure(optimizer, pose, model, log_prefix, **kargs):
    optimizer.zero_grad()
    B = pose.translation.shape[0]
    losses = model(pose, **kargs)
    loss_str = ", ".join(
        ["%s: %.6f" % (k, v.item() / B) for k, v in losses.items()])
    logging.debug("%s %s", log_prefix, loss_str)
    loss = sum(losses.values())
    loss.backward()
    return loss


def optimize_topo(model,
                  pose,
                  cur_iter,
                  nr_step=1000,
                  warmup=100,
                  init_lr=1,
                  minimal_lr=1e-3,
                  init_sigma=1,
                  minimal_sigma=0.1,
                  is_snapshot=False,
                  seq=None,
                  snapshot_dir=None):

    optimizer = torch.optim.Adam(pose.parameters(), lr=init_lr)

    for step in range(nr_step):
        cur_lr = cosine_schedule(
            init_value=init_lr, cur_step=step, nr_step=nr_step,
            warmup=warmup) + minimal_lr
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        cur_sigma = cosine_schedule(init_value=init_sigma,
                                    cur_step=step,
                                    nr_step=nr_step) + minimal_sigma
        model.set_bond_sigma(cur_sigma)
        optimizer.step(
            functools.partial(
                closure,
                optimizer=optimizer,
                model=model,
                pose=pose,
                log_prefix=f"Iter: {cur_iter}, Step: [{step}/{nr_step}],"))
        if is_snapshot:
            coord = pose.to_coord()
            os.makedirs(snapshot_dir, exist_ok=True)
            global_step = (cur_iter - 1) * nr_step + step
            path = os.path.join(snapshot_dir, f"step_{global_step}.pdb")
            write_pdb(
                seq=seq,
                N=coord.N[0].detach().cpu().numpy(),
                CA=coord.CA[0].detach().cpu().numpy(),
                C=coord.C[0].detach().cpu().numpy(),
                CB=coord.CB[0].detach().cpu().numpy(),
                path=path,
            )
