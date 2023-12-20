import torch
from torch.utils.data import DataLoader
import os
import time
import wandb
from utils import Recorder
from tqdm import tqdm
from loguru import logger

class Trainer(object):
    def __init__(self, args, model, dataset, optimizer=None, scheduler=None, collate_fn=None, tokenizer=None, step=0):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = dataset["train"]
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        assert self.collate_fn is not None, "No collate_fn provided."
        
        self.iteration = 0
        self.total_iteration = step
        
        if self.args.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")    
        logger.info('Trainer initialized.')
    
    @Recorder
    def run(self):
        # create model save path
        if not os.path.exists('models/' + self.args.tag):
            os.makedirs('models/' + self.args.tag)
        if not os.path.exists('checkpoints/' + self.args.tag):
            os.makedirs('checkpoints/' + self.args.tag)

        epoch = 0
        # train
        while self.total_iteration < self.args.train_step:

            if self.args.mix_precision:
                loss = self.train_amp()
            else:
                loss = self.train()
            self.save_checkpoint_overwrite()

            if self.args.use_wandb:
                wandb.log({"loss": loss})
            logger.info('Epoch: {}, Loss: {}'.format(epoch, loss))

            epoch += 1

    def train(self):
        self.model.train()
        self.iteration = 0
        train_loss = 0
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn
        )

        accumulate_grad_num = 0

        # train model
        for i, inputs in enumerate(dataloader):
            # get train data
            inputs = self.to_device(inputs)

            out = self.model(**inputs)

            loss = out.loss
            loss = loss / self.args.accumulate_grad_batches  # normalize the loss
            
            loss.backward()
            train_loss += loss.item()

            accumulate_grad_num += 1

            # update model
            if accumulate_grad_num == self.args.accumulate_grad_batches or i == len(dataloader) - 1:

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                accumulate_grad_num = 0

                self.iteration += 1
                self.total_iteration += 1

                if self.total_iteration % self.args.save_step == 0:
                    self.save_checkpoint()

                if self.iteration % self.args.log_step == 0:
                    log_loss = (train_loss * self.args.accumulate_grad_batches) / (i+1)
                    if self.args.use_wandb:
                        wandb.log({"loss": log_loss})
                    logger.info('iteration: {}, loss: {:.8f}, lr * 1e5: {:.4f}, max_gpu_memory: {:.4f} GB'.format(self.iteration, log_loss, self.optimizer.param_groups[0]['lr'] * 1e5, torch.cuda.max_memory_allocated() / 1e9))
                    
        return (train_loss * self.args.accumulate_grad_batches) / len(dataloader)

    def train_amp(self):
        scaler = torch.cuda.amp.GradScaler()
        self.model.train()
        self.iteration = 0
        train_loss = 0
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn
        )

        accumulate_grad_num = 0

        # train model
        for i, inputs in enumerate(dataloader):
            # get train data
            inputs = self.to_device(inputs)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = self.model(**inputs)

                loss = out.loss
                loss = loss / self.args.accumulate_grad_batches  # normalize the loss
                train_loss += loss.item()

            scaler.scale(loss).backward()

            accumulate_grad_num += 1

            # update model
            if accumulate_grad_num == self.args.accumulate_grad_batches or i == len(dataloader) - 1:
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                accumulate_grad_num = 0

                self.iteration += 1
                self.total_iteration += 1

                if self.total_iteration % self.args.save_step == 0:
                    self.save_checkpoint()

                if self.iteration % self.args.log_step == 0:
                    log_loss = (train_loss * self.args.accumulate_grad_batches) / (i+1)
                    if self.args.use_wandb:
                        wandb.log({"loss": log_loss})
                    logger.info('iteration: {}, loss: {:.8f}, lr * 1e5: {:.4f}, max_gpu_memory: {:.4f} GB'.format(self.iteration, log_loss, self.optimizer.param_groups[0]['lr'] * 1e5, torch.cuda.max_memory_allocated() / 1e9))
                    
        return (train_loss * self.args.accumulate_grad_batches) / len(dataloader)

    def to_device(self, inputs):
        if self.args.use_cuda and torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()
        return inputs

    def save_checkpoint(self):
        checkpoint = { 
                    'step': self.total_iteration,
                    'torch_random_state': torch.get_rng_state(),
                    'torch_cuda_random_state': torch.cuda.get_rng_state_all(),
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler,
                    'tokenizer': self.tokenizer
                }
        torch.save(checkpoint, f'./checkpoints/{self.args.tag}/checkpoint_{self.total_iteration}_steps_all.pth')
        torch.save(self.model.state_dict(), f'./models/{self.args.tag}/checkpoint_{self.total_iteration}_steps.pt')