import os
import torch
from torch.utils.data import DataLoader
import wandb
from utils import Recorder
from evaluator import MolT5Evaluator, MolT5Evaluator_cap2smi
from loguru import logger
import pandas as pd

# from data_handler import collate_fn

class Trainer(object):
    def __init__(self, args, model, dataset, optimizer=None, scheduler=None, collate_fn=None, tokenizer=None, epoch=-1):
        self.args = args
        self.model = model
        self.train_evaluator = MolT5Evaluator(args, tokenizer=tokenizer) if self.args.task == 'smiles2cap' else MolT5Evaluator_cap2smi(args, tokenizer=tokenizer)
        self.val_evaluator = MolT5Evaluator(args, tokenizer=tokenizer) if self.args.task == 'smiles2cap' else MolT5Evaluator_cap2smi(args, tokenizer=tokenizer)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collate_fn = collate_fn

        self.epoch = epoch
        self.best_metric = {'train': -1, 'valid': -1}
        
        if self.args.use_cuda and torch.cuda.is_available():
            logger.info('Using GPU...')
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

        # train
        for e in range(self.epoch + 1, self.args.epochs):
            if self.args.mix_precision:
                loss = self.train_amp()
            else:
                loss = self.train()
            if self.args.use_wandb:
                wandb.log({"loss": loss})
            logger.info('Epoch: {}, Loss: {:.8f}'.format(e, loss))

            if (e + 1) % self.args.save_epoch == 0:
                self.save_checkpoint(e)

            if (e + 1) % self.args.eval_epoch == 0:
                self.validate()
                
        self.test()

    # Training code without mixed precision, but we almost always use mixed precision in our experiments.
    def train(self):

        self.model.train()
        iteration = 0
        train_loss = 0
        dataloader = DataLoader(
            self.dataset["train"],
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

                iteration += 1

                if iteration % self.args.log_step == 0:
                    log_loss = (train_loss * self.args.accumulate_grad_batches) / (i+1)
                    if self.args.use_wandb:
                        wandb.log({"loss": log_loss})
                    logger.info('iteration: {}, loss: {:.8f}, lr * 1e5: {:.4f}, max_gpu_memory: {:.4f} GB'.format(iteration, log_loss, self.optimizer.param_groups[0]['lr'] * 1e5, torch.cuda.max_memory_allocated() / 1e9))

        return (train_loss * self.args.accumulate_grad_batches) / len(dataloader)
    
    # Training code with mixed precision, all same as train() function except using mixed precision.
    def train_amp(self):
        scaler = torch.cuda.amp.GradScaler()
        self.model.train()
        iteration = 0
        train_loss = 0
        dataloader = DataLoader(
            self.dataset["train"],
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

                iteration += 1

                if iteration % self.args.log_step == 0:
                    log_loss = (train_loss * self.args.accumulate_grad_batches) / (i+1)
                    if self.args.use_wandb:
                        wandb.log({"loss": log_loss})
                    logger.info('iteration: {}, loss: {:.8f}, lr * 1e5: {:.4f}, max_gpu_memory: {:.4f} GB'.format(iteration, log_loss, self.optimizer.param_groups[0]['lr'] * 1e5, torch.cuda.max_memory_allocated() / 1e9))

        return (train_loss * self.args.accumulate_grad_batches) / len(dataloader)

    def validate(self):
        tag = 'valid'
        metrics = self.evaluation(tag)
        # log metrics
        self.print_metrics(metrics, status=tag)
        self.store_best_model(metrics['bleu4'], tag=tag)

    def test(self, final_test=False):
        tag = 'test'
        logger.info('Start testing using current model.')
        metrics = self.evaluation(tag)
        self.print_metrics(metrics, status=tag)

        logger.info('Start testing using best validation model.')
        self.get_best_model()
        metrics = self.evaluation(tag)
        self.print_metrics(metrics, status=tag)

    @torch.no_grad()
    def evaluation(self, dataset):
        # set model to eval mode
        self.model.eval()

        dataloader = DataLoader(
            self.dataset[dataset],
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.eval_num_workers,
            collate_fn=self.collate_fn
        )

        # reset evaluator
        self.val_evaluator.reset()
        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                inputs = self.to_device(inputs)
                # delete labels from inputs
                labels = inputs['labels']
                del inputs['labels']
                out = self.model.generate(**inputs, num_beams=1, max_new_tokens=self.args.max_new_tokens)
                # store results
                self.val_evaluator(out, labels)
            # compute metrics
            metrics = self.val_evaluator.evaluate()
            return metrics

    def output_case(self):
        self.model.eval()
        dataloader = DataLoader(
            self.dataset['test'],
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.eval_num_workers,
            collate_fn=self.collate_fn
        )
        input_list = []
        pred_list = []
        label_list = []
        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                inputs = self.to_device(inputs)
                # delete labels from inputs
                labels = inputs['labels']
                del inputs['labels']
                out = self.model.generate(**inputs, num_beams=1, max_new_tokens=self.args.max_new_tokens)
                pred_list.extend([self.tokenizer.decode(x, skip_special_tokens=True) for x in out])
                labels[labels == -100] = 0
                label_list.extend([self.tokenizer.decode(x, skip_special_tokens=True) for x in labels])
                input_list.extend([self.tokenizer.decode(x, skip_special_tokens=True) for x in inputs['input_ids']])
        info = list(zip(input_list, pred_list, label_list))
        df = pd.DataFrame(info, columns=['input', 'pred', 'label'])
        df.to_csv("./case_study/" + self.args.tag + '.csv', header=None, index=False)
    
    def print_metrics(self, metrics, status='train'):
        status = status.capitalize()
        logger.info('{}: Metrics: '.format(status) + ', '.join(['{}: {:.8f}'.format(k, v) for k, v in metrics.items()]))
        if self.args.use_wandb:
            for k, v in metrics.items():
                wandb.log({f"{status}_" + k: v})
    
    def to_device(self, inputs):
        if self.args.use_cuda and torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()
        return inputs

    def get_best_model(self):
        self.model.load_state_dict(torch.load('models/' + self.args.tag + '/best_valid.pt'))

    def store_best_model(self, metric, tag):
        cur_metric = self.best_metric[tag]
        if metric > cur_metric:
            self.best_metric[tag] = metric
            torch.save(self.model.state_dict(), 'models/' + self.args.tag + '/best_' + tag + '.pt')

    def save_checkpoint(self, epoch):
        checkpoint = { 
                    'epoch': epoch,
                    'torch_random_state': torch.get_rng_state(),
                    'torch_cuda_random_state': torch.cuda.get_rng_state_all(),
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler,
                    'tokenizer': self.tokenizer
                }
        torch.save(checkpoint, f'./checkpoints/{self.args.tag}/checkpoint_{epoch}_epochs_all.pth')

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f'./models/{self.args.tag}/{epoch}_epochs.pt')