# The calculation of FCD metric in molecule generation task requires tensorflow.
# Our code is based on pytorch, and the cuda version and cudnn version are not compatible with tensorflow.
# Thus, we need to disable GPU for tensorflow, only using CPU to calculate FCD metric, or it will cause unexpected error.
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
cpus = tf.config.experimental.list_physical_devices('CPU')
tf.config.experimental.set_visible_devices(cpus, 'CPU')
assert not tf.config.experimental.list_physical_devices('GPU')
del os.environ["CUDA_VISIBLE_DEVICES"]

import wandb
from trainer import Trainer
from data_handler import get_data, collate_fn
from utils import seed_everything, init_wandb, load_checkpoint
from utils import get_model_and_tokenizer, get_optimizer_and_scheduler
from myparser import arg_parser


def main(args):
    # load all data
    dataset = get_data(args)

    # if load checkpoint to resume training, load all states
    if args.load_checkpoint:
        checkpoint = load_checkpoint(args)
        torch.set_rng_state(checkpoint['torch_random_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'][0])
        epoch = checkpoint['epoch']
        if args.epochs is None:
            actual_batch_size = args.batch_size * args.accumulate_grad_batches
            total_steps = args.steps
            args.epochs = int(total_steps * actual_batch_size / len(dataset['train'])) + 1
        model = checkpoint['model']
        tokenizer = checkpoint['tokenizer']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

    # train from scratch
    else:
        epoch = -1
        seed_everything(args.seed)
        model, tokenizer = get_model_and_tokenizer(args)
        optimizer, scheduler = get_optimizer_and_scheduler(model, args, len(dataset['train']), warmup_steps=args.warmup)
    
    # initialize trainer and start training
    trainer = Trainer(args, model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler, collate_fn=collate_fn, tokenizer=tokenizer, epoch=epoch)
    trainer.run()

if __name__ == '__main__':
    args = arg_parser()
    if args.use_wandb:
        init_wandb(args)
    main(args)
    if args.use_wandb:
        wandb.finish()