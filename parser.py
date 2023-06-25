import argparse
import json


def parse_arg_train():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument(
        '--dataset',
        help='Dataset to use',
        choices=['msvd-qa-oe', 'msrvtt-qa-oe', 'tgif-frameqa', 'tgif-count', 'tgif-action', 'tgif-transition'],
        type=str,
        required=True,
    )
    parser.add_argument('--dataset-dir', help='Directory path to dataset for train and validation', required=True)

    parser.add_argument('--log-dir', help='Log directory', default='./runs')

    parser.add_argument('--ckpt-interval', help='How many epoch between checkpoints', default=1, type=int)
    parser.add_argument('--model-path', help='Load pretrained model')

    parser.add_argument('--batch-size', help='Batch size for training', default=20, type=int)
    parser.add_argument('--eval-per-epoch', help='Total validation per epoch', default=1, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
    parser.add_argument('--drop-out-rate', help='Drop out rate for training', default=0.5, type=float)
    parser.add_argument('--lr', help='Learning rate for training', nargs='+', default=[5e-6], type=float)
    parser.add_argument('--min-lr', help='Minimum learning rate after decaying', default=1e-8, type=float)
    parser.add_argument(
        '--temporal-scale',
        help='Scales for multisegment sampling',
        nargs='+',
        default=[1, 2, 3],
        type=int,
    )
    parser.add_argument(
        '--patience',
        help='Number of stagnant epoch before decay (only for reduce on plateau scheduler)',
        default=0.5,
        type=int,
    )
    parser.add_argument(
        '--lr-decay-factor',
        help='Learning rate decay factor (after full-cycle for cosine scheduler)',
        default=0.5,
        type=float,
    )
    parser.add_argument(
        '--lr-warm-up',
        help='Percentage of epoch to do linear warmup [0,1)',
        default=0.1,
        type=float,
    )
    parser.add_argument(
        '--lr-restart-epoch',
        help='Number of epoch before restarting the learning rate (only for cosine annealing scheduler)',
        default=2,
        type=int,
    )
    parser.add_argument(
        '--lr-restart-mul',
        help='Multiplier for lr-restart-epoch after restart (only for cosine annealing scheduler)',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--use-cosine-scheduler',
        help='Whether to use cosine annealing scheduler or reduce on plateau scheduler',
        action='store_true',
    )

    parser.add_argument('--reg-strength', help='Weight for L2 regularization', default=0.001, type=float)
    parser.add_argument('--num-workers', help='Number of workers for dataloader', default=2, type=int)

    parser.add_argument(
        '--use-hinge-loss',
        help='Use hinge loss instead of cross entropy (for mc task)',
        action='store_true',
    )
    parser.add_argument('--margin', help='Margin for hingle loss (only for mc task)', default=1, type=float)

    parser.add_argument(
        '--debug-mode',
        help='If on, it will not write logs and checkpoints',
        action='store_true',
    )
    parser.add_argument(
        '--sanity-check',
        help='Sanity check by overfitting model with very small dataset',
        action='store_true',
    )
    parser.add_argument('--comment', help='Additional comment if needed', default='', type=str)

    result = parser.parse_args()

    if result.use_cosine_scheduler:
        del vars(result)['patience']
    else:
        del vars(result)['lr_restart_epoch']
        del vars(result)['lr_restart_mul']
        del vars(result)['lr_warm_up']

    if not result.use_hinge_loss:
        del vars(result)['margin']

    if result.comment == '':
        del vars(result)['comment']

    model_args = json.load(open(f'configs/{result.dataset}.json', 'r'))
    vars(result).update(model_args)

    if len(result.lr) == 1:
        result.lr = result.lr * 3

    if len(result.temporal_scale) < 1:
        result.temporal_scale = [1, 2, 3]
    return result