import argparse

import torch
import numpy as np
import random
import os

from embgen_strategy.fix_m0 import fix_m0
from embgen_strategy.nobc import nobc
from embgen_strategy.joint_sloss import joint_sloss
from embgen_strategy.joint_mloss import joint_mloss
from embgen_strategy.posthoc_sloss import posthoc_sloss
from embgen_strategy.posthoc_mloss import posthoc_mloss
from embgen_strategy.finetune_m0 import finetune_m0

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PinSAGE')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--strategy', type=str, default='train-once',
                        help='strategy to generate node embeddings on evolving graphs')
    parser.add_argument('--dataset', type=str, default='Amazon-Musical_Instruments',
                        help='dataset name (default: Amazon-Musical_Instruments)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 2048)')  
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay (default: 1e-2)')                    
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--log_every', type=int, default=10,
                        help='how often we wanna evaluate the performance (default: 10)')                        
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_path', type=str, default="",
                        help='checkpoint path')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    args = parser.parse_args()

    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    if os.path.exists(args.checkpoint_path):
        # if args.strategy != 'train-once':
        print('checkpoint already exists. exit.')
        exit(-1)

    args.trans_type = None
    if 'notrans' in args.strategy:
        args.trans_type = 'notrans'
    elif 'linear' in args.strategy:
        args.trans_type = 'linear'

    if args.strategy == 'fix-m0':
        '''
            Train M0 on G0 once, and use the same GNN over G1, G2, ...
        '''
        assert args.trans_type is None

        fix_m0(args)

    elif 'nobc' in args.strategy:
        '''
            Train Mk on Gk independently without backward compatibility
        '''
        assert args.trans_type is None

        nobc(args)
        

    elif 'posthoc' in args.strategy:
        '''
            Aligning models trained by nobc.

            posthoc-linear-sloss
            posthoc-linear-mloss
        '''
        assert args.trans_type == 'linear'
        args.epochs = 200

        dir_name = os.path.dirname(args.checkpoint_path)
        args.checkpoint_vanilla_path = os.path.join(dir_name, 'nobc')

        if not os.path.exists(args.checkpoint_vanilla_path):
            raise RuntimeError(f'Cannot find train-from-scratch-vanilla checkpoint at {args.checkpoint_vanilla_path}')

        if 'mloss' in args.strategy:
            posthoc_mloss(args)
        elif 'sloss' in args.strategy:
            posthoc_sloss(args)

    elif 'joint-' in args.strategy:
        '''
            joint-linear-sloss-lam16
            joint-notrans-sloss-lam16

            joint-linear-mloss-lam16
        '''
        tmp = args.strategy.split('-')
        args.lam = float(tmp[-1][3:])

        assert args.trans_type is not None
            
        if '-sloss' in args.strategy:
            joint_sloss(args)
        elif '-mloss' in args.strategy:
            assert args.trans_type == 'linear'
            joint_mloss(args)
        else:
            raise ValueError('Invalid strategy name.')

    elif 'finetune-m0' in args.strategy:
        finetune_m0(args)

    else:
        raise ValueError(f'Unknown embedding generation strategy called {args.strategy}.')

if __name__ == "__main__":
    main()