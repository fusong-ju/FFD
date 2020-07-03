import os
import sys
import logging
import torch
from utils.io_tool import read_fasta, write_pdb
from model.backbone_model import BackBoneModel
from model.coordinates import ResiduePose
from .optimizer import optimize_topo


class Task:
    def __init__(self,
                 fasta_path,
                 prediction,
                 outdir,
                 n_structs,
                 pool_size,
                 device,
                 is_snapshot=False):
        self.init_logger(os.path.join(outdir, "log.txt"))
        self.name = os.path.splitext(os.path.basename(fasta_path))[0]
        self.seq = read_fasta(fasta_path)
        self.L = len(self.seq)

        self.model = BackBoneModel(
            pred_dist=prediction.cbcb,
            pred_omega=prediction.omega,
            pred_theta=prediction.theta,
            pred_phi=prediction.phi,
        ).to(device)
        logging.info(str(self.model))
        self.model.set_default_weight()

        self.outdir = outdir
        self.n_structs = n_structs
        self.device = device
        self.is_snapshot = is_snapshot

        self.pool = [None] * pool_size
        self.score = [None] * pool_size
        self.n_iter = [None] * pool_size
        self.n_finished = 0

    def init_logger(self, path):
        logging.getLogger().setLevel(logging.DEBUG)
        fd = logging.FileHandler(path)
        logging.getLogger().addHandler(fd)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    def start(self):
        cur_iter = 0
        while self.n_finished < self.n_structs:
            cur_iter += 1
            self.before_iteration()
            poses = self.get_poses()
            optimize_topo(
                self.model,
                poses,
                cur_iter,
                is_snapshot=self.is_snapshot,
                seq=self.seq,
                snapshot_dir=os.path.join(self.outdir, "snapshot"),
            )
            self.after_iteration(poses)
            del poses

    def before_iteration(self):
        z = self.L**(1.0 / 3) * 4
        for i in range(len(self.pool)):
            if self.pool[i] is None:
                rand_x = torch.rand(self.L, 3) * z
                rand_y = 2 * torch.rand(self.L, 4) - 1
                self.pool[i] = (rand_x.to(self.device), rand_y.to(self.device))
                self.score[i] = None
                self.n_iter[i] = 0
                logging.info(f"Pool {i} generate new structure.")

    def get_poses(self):
        x = torch.stack([_[0] for _ in self.pool])
        y = torch.stack([_[1] for _ in self.pool])
        return ResiduePose(x, y)

    def score_it(self, translation, quaternion):
        with torch.no_grad():
            x = ResiduePose(translation[None], quaternion[None])
            score = self.model(x)
            del x
        return score

    def after_iteration(self, poses):
        with torch.no_grad():
            translation = poses.translation.data
            quaternion = poses.quaternion.data
            for i in range(len(self.pool)):
                item = (translation[i], quaternion[i])
                score = self.score_it(*item)
                s_score = sum(score.values()).item()
                score_str = ", ".join(
                    ["%s: %f" % (k, v.item()) for k, v in score.items()])
                score_str += f", Total: {s_score}"
                logging.info(f"Pool {i} score, {score_str}")
                if self.n_iter[i] > 0 and self.score[i] * 1.001 < s_score:
                    logging.info(
                        f"Pool {i} auto converge at iteration {self.n_iter[i]}. "
                    )
                    self.save_decoy(*self.pool[i], info=score_str)
                    self.pool[i] = None
                else:
                    self.pool[i] = item
                    self.score[i] = s_score
                    self.n_iter[i] += 1

    def save_decoy(self, translation, quaternion, info=None):
        self.n_finished += 1
        pose = ResiduePose(translation[None], quaternion[None])
        coord = pose.to_coord()

        decoy_dir = os.path.join(self.outdir, "decoys")
        os.makedirs(decoy_dir, exist_ok=True)
        path = os.path.join(decoy_dir, f"{self.name}_{self.n_finished}.pdb")
        write_pdb(
            seq=self.seq,
            N=coord.N[0].cpu().numpy(),
            CA=coord.CA[0].cpu().numpy(),
            C=coord.C[0].cpu().numpy(),
            CB=coord.CB[0].cpu().numpy(),
            path=path,
            info=info
        )
        del coord, pose
        logging.info(f"Save decoy {self.n_finished} to {path}")
