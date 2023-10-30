# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import torch
import mmcv
from mmcv import DictAction
from PIL import Image
from tqdm import tqdm

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_unconditional_model  # isort:skip  # noqa


# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Generation demo')
    parser.add_argument('--config',
                        default='/home/ivms/net_disk_project/19045845/dataclean/mmgeneration_add/results/result_stylegan2_c2_ada_256_b4x8_30k_furniterbg/stylegan2_c2_ada_256_b4x8_30k.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='/home/ivms/net_disk_project/19045845/dataclean/mmgeneration_add/results/result_stylegan2_c2_ada_256_b4x8_30k_furniterbg/ckpt/result_stylegan2_c2_ada_256_b4x8_30k_furniterbg/iter_276500.pth'
                        , help='checkpoint file')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos',
        help='path to save unconditional samples')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')

    # args for inference/sampling
    parser.add_argument(
        '--num-batches', type=int, default=4, help='Batch size in inference')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=12,
        help='The total number of samples')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='Which model to use for sampling')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    # args for image grid
    parser.add_argument(
        '--padding', type=int, default=0, help='Padding in the image grid.')
    parser.add_argument(
        '--nrow',
        type=int,
        default=6,
        help='Number of images displayed in each row of the grid')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    for batch, i in tqdm(enumerate(range(0, int(5000 / args.num_samples)))):
        results = sample_unconditional_model(model, args.num_samples,
                                             args.num_batches, args.sample_model,
                                             **args.sample_cfg)
        results = (results[:, [2, 1, 0]] + 1.) / 2.

        # save images
        mmcv.mkdir_or_exist(os.path.dirname(args.save_path))

        #     import pdb;pdb.set_trace()
        for index, img in enumerate(results):
            img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(img)
            im.save(os.path.join(args.save_path, f'{batch}_{index}.png'))

            # utils.save_image(
            #             results, args.save_path, nrow=args.nrow, padding=args.padding)


if __name__ == '__main__':
    main()
