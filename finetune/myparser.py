# We use this file to parse the arguments from command line
def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="Tag")

    # basic training information
    parser.add_argument("--train_path", type=str, default="./data/ChEBI-20/train.json")
    parser.add_argument("--test_path", type=str, default="./data/ChEBI-20/dev.json")
    parser.add_argument("--valid_path", type=str, default="./data/ChEBI-20/test.json")
    parser.add_argument("--model", type=str, default="SCIR-HI/ada-t5-small")
    parser.add_argument("--task", type=str, default="smiles2cap", help="smiles2cap or cap2smiles", choices=["smiles2cap", "cap2smiles"])

    # loading information
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)

    # training hyperparameters and strategies
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--mix_precision", action="store_true", default=False)

    # others
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default="./model/")
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--eval_epoch", type=int, default=20)
    parser.add_argument("--save_epoch", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--eval_num_workers", type=int, default=32)
    
    # just for jupyter notebook
    parser.add_argument("-f", type=str, default='')
    return parser.parse_args()