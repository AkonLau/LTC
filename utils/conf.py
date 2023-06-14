import os
import argparse
import pprint
import yaml
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict


parser = argparse.ArgumentParser(description='PyTorch Training')

def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    # arg_lists.append(arg)
    return arg

# ------------------------------
def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    # print("Config:\n" + pprint.pformat(cfg))
    return edict(cfg)


# ------------------------------
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        #if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            #if k not in b:
            b[k] = v

def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def cfg_from_file(cfg):
    """Load a config from file filename and merge it into the default options.
    """
    filename=cfg.config
    # args from yaml file
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, cfg)

    return cfg


def get_config():
    # args from argparser
    cfg = parser2dict()
    cfg = cfg_from_file(cfg)
    if 'mixmethod' in cfg:
        cfg['mixmethod'] = cfg['mixmethod'].split(',')
        if len(cfg['mixmethod']) == 1:
            cfg['mixmethod'] = cfg['mixmethod'][0]

    if cfg.dataset == 'iNaturalist18' or cfg.dataset == 'Imagenet-LT':
        cfg['pretrained'] = 0
        cfg['lr_group'] = [cfg.lr, cfg.lr]
        cfg['lrstep'] = 'none'
        cfg['lrscheduler'] = 'CosineAnnealingLR'
    else:
        if not cfg.pretrained:
            cfg['lr_group'] = [0.1, 0.1]
            cfg['epochs'] = 200
            cfg['lrstep'] = [100, 150]
        if cfg['epochs'] == 300:
            cfg['lrstep'] = [150, 225, 270]
        if cfg['epochs'] == 100:
            cfg['lrstep'] = [40, 70]

    if cfg.dataset in ['nabirds','cub']:
        cfg['warp'] = False
    return cfg


def set_env(cfg):
    # set seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed) # cpu vars
    torch.manual_seed(cfg.seed) # cpu  vars
    torch.cuda.manual_seed(cfg.seed) # gpu  vars
    torch.cuda.manual_seed_all(cfg.seed) # gpu vars
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False

    cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

# ----------------------------------------------------------------------------------------
# base
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--weightfile', default=None, type=str, metavar='PATH', help='path to model (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')
parser.add_argument('--config', default='config/comm.yml', type=str, help='config files')

# train
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--pretrained', default=1, type=float, help='loss weights')

# other
parser.add_argument('--mixmethod', default='baseline', type=str, help='config files')
parser.add_argument('--netname', default='resnet50', type=str, help='config files')
parser.add_argument('--prob', type=float, default=1.0, help='')
parser.add_argument('--beta', type=float, default=1.0, help='')
parser.add_argument('--dataset', default='cub', type=str, help='dataset')
parser.add_argument('--cropsize', default=448, type=int, metavar='N', help='cropsize')
parser.add_argument('--midlevel', dest='midlevel', action='store_true', help='midlevel')
parser.add_argument('--train_proc', default='comm', type=str, help='dataset')
parser.add_argument('--start_eval', default=-1, type=int, metavar='N', help='network depth')
parser.add_argument('--iteral', type=int, default=1000, help='iteral for save_checkpoint')

# Auxiliary target coding regularization
parser.add_argument('--LTC', dest='LTC', action='store_true', help='learnable target coding loss')
parser.add_argument('--HTC', dest='HTC', action='store_true', help='hadamard target coding loss')

parser.add_argument('--gamma_', type=float, default=1, help="parameter for mse loss")
parser.add_argument('--lambda_', type=float, default=0.01, help="parameter for triplet loss")
parser.add_argument('--beta_', type=float, default=0.1, help="parameter for distance loss")

parser.add_argument('--code_length', type=int, default=512, help="code_length for auxiliary target codes")
parser.add_argument('--active_type', type=str, default='sgn', help='loss type: sgn or tanh')
parser.add_argument('--margin_ratio', type=float, default=1, help="margin_ratio for triplet loss")

# just for imbalanced data learning
parser.add_argument('--train_rule', type=str, default='None', help="train_rule for iNaturalist18 training: None or DRW")