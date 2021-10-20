"""Train script."""
import os
import logging

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import ttools

from marionet import datasets, models, callbacks
from marionet.interfaces import Interface

LOG = logging.getLogger(__name__)

th.backends.cudnn.deterministic = True


def _worker_init_fn(_):
    np.random.seed()


def _set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def main(args):
    LOG.info(f"Using seed {args.seed}.")

    device = "cuda" if th.cuda.is_available() and args.cuda else "cpu"

    learned_dict = models.Dictionary(args.num_classes,
                                     (args.canvas_size // args.layer_size*2,
                                      args.canvas_size // args.layer_size*2),
                                     4, bottleneck_size=args.dim_z)
    learned_dict.to(device)

    model = models.Model(learned_dict, args.layer_size, args.num_layers)
    model.eval()

    model_checkpointer = ttools.Checkpointer(
        os.path.join(args.checkpoint_dir, "model"), model)
    model_checkpointer.load_latest()

    with th.no_grad():
        fwd_data = model(im, None, hard=True)


if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()

    # Representation
    parser.add_argument("--layer_size", type=int, default=8,
                        help="size of anchor grid")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of layers")
    parser.add_argument("--num_classes", type=int, default=150,
                        help="size of dictioanry")

    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
