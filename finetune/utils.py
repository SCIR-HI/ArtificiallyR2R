import torch
import wandb
import time
import os
from torch.optim import AdamW
from loguru import logger
from transformers import get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer

# record the running time of a function
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

# initialize wandb
@Recorder
def init_wandb(args):
    while True:
        try:
            os.environ["WANDB_SILENT"] = "true"
            args_dict = vars(args)
            wandb.init(
                project='my project',
                name=args.tag,
                config=args_dict
            )
            return
        except:
            logger.info('Wandb initialization failed.')

# initialize model and tokenizer from transformers
@Recorder
def get_model_and_tokenizer(args):
    while True:
        try:
            model_name = args.model
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
            logger.info(f'Model and tokenizer loaded from {model_name}.')
            break
        except Exception as e:
            # sometimes may encounter network error
            logger.info(f'Exception: {e}')
            logger.info('Retrying...')
            continue
    return model, tokenizer

# set random seed
# @Recorder
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# initialize optimizer and scheduler
@Recorder
def get_optimizer_and_scheduler(model, args, step, warmup_steps):

    # initialize optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)


    # when start a training, we only specify either steps or epochs
    assert args.steps is None or args.epochs is None, 'Please specify either steps or epochs, not both.'
    actual_batch_size = args.batch_size * args.accumulate_grad_batches
    # if steps is specified, automatically calculate epochs because training code requires epochs
    if args.steps is not None:
        total_steps = args.steps
        args.epochs = int(total_steps * actual_batch_size / step) + 1
    else:
        total_steps = int(step * args.epochs / actual_batch_size)
        
    # initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    # log taining parameters
    logger.info(f"Single training step: {step / (args.batch_size * args.accumulate_grad_batches)}")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Accumulate gradient batches: {args.accumulate_grad_batches}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {args.lr}")
    return optimizer, scheduler

# load model state dict from trained model parameters
@Recorder
def load_model(model, args):
    if args.load_model:
        if args.load_path is None:
            logger.warning('No model loaded.')
        else:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(args.load_path))
            else:
                model.load_state_dict(torch.load(args.load_path, map_location=torch.device('cpu')))
            logger.info(f'Model loaded from {args.load_path}.')
    return model

# load while checkpoint, including:
# 1. model state dict
# 2. optimizer state dict
# 3. scheduler state dict
# 4. training epoch
# 5. random seed state
# 6. random seed state of cuda
# 7. tokenizer
# This function is used for resuming training
@Recorder
def load_checkpoint(args):
    assert args.load_path is not None, 'Please specify the path of checkpoint.'
    checkpoint = torch.load(args.load_path)
    logger.info(f"Loaded checkpoint from {args.load_path}")
    return checkpoint
