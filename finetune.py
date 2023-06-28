import torch
import lightning.pytorch as pl

from lseg.modules.lseg_module import LSegModule
from .lseg.utils import make_checkpoint_callbacks, get_wandb_logger
from model import Model
from options import Options

args = Options().parse()

torch.manual_seed(args.seed)
args.test_batch_size = 1 
alpha=0.5
    
args.scale_inv = False
args.widehead = True
args.dataset = 'ade20k'
args.backbone = 'clip_vitl16_384'
args.weights = 'checkpoints/demo_e200.ckpt'
args.ignore_index = 255
args.data_path = '../datasets'
args.exp_name = 'lseg_ade20k_l16'

lseg = LSegModule.load_from_checkpoint(
    checkpoint_path=args.weights,
    data_path=args.data_path,
    dataset=args.dataset,
    backbone=args.backbone,
    aux=args.aux,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0.004,
    batch_size=1,
    max_epochs=240,
    ignore_index=args.ignore_index,
    dropout=0.0,
    scale_inv=args.scale_inv,
    augment=False,
    no_batchnorm=False,
    widehead=args.widehead,
    widehead_hr=args.widehead_hr,
    map_locatin="cpu",
    arch_option=0,
    block_depth=0,
    activation='lrelu',
)

model = Model(lseg)

args.gpus = -1
args.accelerator = "ddp"
args.benchmark = True
args.version = 0
args.sync_batchnorm = True

ttlogger = pl.loggers.TestTubeLogger(
    "checkpoints", name=args.exp_name, version=args.version
)

args.callbacks = make_checkpoint_callbacks(args.exp_name, args.version)

wblogger = get_wandb_logger(args)
args.logger = [wblogger, ttlogger]

trainer = pl.Trainer.from_argparse_args(args)
trainer.fit(model)