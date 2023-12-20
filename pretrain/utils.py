import torch
import wandb
import time
import os
from torch.optim import AdamW
from loguru import logger
from transformers import get_linear_schedule_with_warmup, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer

def Recorder(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Started execution of {func.__name__}")
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Finished execution of {func.__name__} in {elapsed_time:.6f} seconds")
        return result
    return wrapper

@Recorder
def init_wandb(args):
    cnt = 0
    while True:
        try:
            os.environ["WANDB_SILENT"] = "true"
            args_dict = vars(args)
            wandb.init(
                project='My project',
                name=args.tag,
                config=args_dict
            )
            return
        except:
            logger.info('Wandb initialization failed. Retrying...')
            cnt += 1
            if cnt > 10:
                raise Exception('Wandb initialization failed too many times.')

@Recorder
def get_model_and_tokenizer(args):
    cnt = 0
    while True:
        try:
            model_name = args.model
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
            break
        except Exception as e:
            logger.info(f'Exception: {e}. Retrying...')
            cnt += 1
            if cnt > 10:
                raise Exception('Model and tokenizer initialization failed too many times.')
    return model, tokenizer

@Recorder
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@Recorder
def get_optimizer_and_scheduler(model, args, step, warmup_steps=None):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.train_step
    if warmup_steps is None:
        warmup_steps = args.warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    logger.info(f"Training step per epoch: {step / (args.batch_size * args.accumulate_grad_batches)}")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {args.lr}")
    return optimizer, scheduler

@Recorder
def load_checkpoint(args):
    assert args.load_path is not None, 'Please specify the path of checkpoint.'
    checkpoint = torch.load(args.load_path)
    logger.info(f"Loaded checkpoint from {args.load_path}")
    return checkpoint
    
