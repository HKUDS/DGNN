import os, argparse
from yaml import safe_load as yaml_load
from json import dumps as json_dumps

def parse_args(show_args=True):
    parser = argparse.ArgumentParser(description='HGMN Arguments')
    parser.add_argument('--desc', type=str, default='')

    """ Configuration Arguments """
    parser.add_argument('--cuda',   type=str, default='0')
    parser.add_argument('--seed',   type=int, default=2020)

    """ Model Arguments """
    parser.add_argument('--n_hid',     type=int,   default=16)
    parser.add_argument('--n_layers',  type=int,   default=2)
    parser.add_argument('--mem_size',  type=int,   default=8)
    # parser.add_argument('--fuse',      type=str,   default=None, help='mean or weight')

    """ Train Arguments """
    parser.add_argument('--dropout', type=float, default=0)

    """ Optimization Arguments """
    parser.add_argument('--lr',         type=float, default=0.01)
    parser.add_argument('--reg',        type=float, default=0.0005)
    parser.add_argument('--decay',      type=float, default=0.985)
    parser.add_argument('--decay_step', type=int,   default=1)
    parser.add_argument('--n_epoch',    type=int,   default=150)
    parser.add_argument('--batch_size', type=int,   default=4096)
    parser.add_argument('--patience',   type=int,   default=30)

    """ Valid/Test Arguments """
    parser.add_argument('--topk',            type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=4096)

    """ Data Arguments """
    parser.add_argument('--dataset',       type=str, default="")
    parser.add_argument('--data_path',     type=str, default="")
    parser.add_argument('--val_neg_path',  type=str, default="")
    parser.add_argument('--test_neg_path', type=str, default="")
    parser.add_argument('--num_workers',   type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=None)
    parser.add_argument('--checkpoint',    type=str, default="")
    parser.add_argument('--model_dir',     type=str, default="")

    args = parser.parse_args()

    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if show_args:
        print('Aruments:\n{}'.format(json_dumps(args.__dict__, indent=4)), flush=True)

    return args
