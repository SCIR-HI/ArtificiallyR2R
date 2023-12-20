# disable GPU for tensorflow, or it will cause unexpected error
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
from data_handler import collate_fn, get_test_data
from utils import seed_everything, init_wandb, load_model
from utils import get_model_and_tokenizer
from myparser import arg_parser


def main(args):
    dataset = get_test_data(args)
    seed_everything(args.seed)
    model, tokenizer = get_model_and_tokenizer(args)
    model = load_model(model, args)
    trainer = Trainer(args, model=model, dataset=dataset, collate_fn=collate_fn, tokenizer=tokenizer)
    trainer.test()

if __name__ == '__main__':
    args = arg_parser()
    if args.use_wandb:
        init_wandb(args)
    main(args)
    if args.use_wandb:
        wandb.finish()