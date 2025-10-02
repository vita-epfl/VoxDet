import torch
from argparse import ArgumentParser


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--source_path', default='/mnt/vita/scratch/vita-students/users/wuli/code/CGFormer/pretrain/tensorboard/version_0/checkpoints/last2.ckpt')
    parser.add_argument('--dst_path', default='/mnt/vita/scratch/vita-students/users/wuli/code/CGFormer/pretrain/tensorboard/version_0/checkpoints/last.ckpt')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_config()

    checkpoints = torch.load(args.source_path, map_location='cpu')['state_dict']
    new_checkpoints = {}
    for key in checkpoints:
        new_checkpoints[key.replace('model.', '')] = checkpoints[key]
    
    torch.save({'state_dict': new_checkpoints}, args.dst_path)