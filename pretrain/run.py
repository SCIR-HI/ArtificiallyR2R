import torch
from trainer import Trainer
from data_handler import get_data, collate_fn
from utils import seed_everything, init_wandb, load_checkpoint
from utils import get_model_and_tokenizer, get_optimizer_and_scheduler
from myparser import arg_parser


def main(args):
    dataset = get_data(args)
    if args.load_checkpoint:
        checkpoint = load_checkpoint(args)
        step = checkpoint['step']
        model = checkpoint['model']
        tokenizer = checkpoint['tokenizer']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        torch.set_rng_state(checkpoint['torch_random_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'][0])
    else:
        step = 0
        model, tokenizer = get_model_and_tokenizer(args)
        optimizer, scheduler = get_optimizer_and_scheduler(model, args, len(dataset['train']), warmup_steps=1000)    
    trainer = Trainer(args, model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler, collate_fn=collate_fn, tokenizer=tokenizer, step=step)
    trainer.run()

if __name__ == '__main__':
    args = arg_parser()
    if args.use_wandb:
        init_wandb(args)
    seed_everything(args.seed)
    main(args)