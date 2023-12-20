import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="Tag")

    # Initialization information for training
    parser.add_argument("--train_path", type=str, default="./data/train.json")
    parser.add_argument("--model", type=str, default="google/t5-v1_1-small")

    # Hyperparameters for training
    parser.add_argument("--batch_size", type=int, default=64) # Note that the actual batch size is (batch_size * accumulate_grad_batches)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2) # set to 1 if you don't want to use gradient accumulation
    parser.add_argument("--train_step", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--mix_precision", action="store_true", default=False)

    # Others
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--save_step", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)

    # Only useful when you want to train from a breakpoint
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    
    # just for jupyter notebook
    parser.add_argument("-f", type=str, default='')
    return parser.parse_args()