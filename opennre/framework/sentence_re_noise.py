import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader, SentenceRELoader_Noise
from .utils import AverageMeter
import json

class SentenceRENoise(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 logger,
                 batch_size=32,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 warmup_step=300,
                 opt='sgd',
                 noise_rate=None,
                 noise_mode=None
                 ):

        super().__init__()
        self.max_epoch = max_epoch
        self.logger = logger
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader_Noise(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                noise_mode=noise_mode,
                noise_rate=noise_rate)

        if val_path != None:
            self.val_loader = SentenceRELoader_Noise(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                noise_mode=noise_mode,
                noise_rate=noise_rate)

        if test_path != None:
            self.test_loader = SentenceRELoader_Noise(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                noise_mode=noise_mode,
                noise_rate=noise_rate)
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()  # cross entropy loss 中集成有softmax
        self.elem_criterion = nn.CrossEntropyLoss(reduction='none')  # cross entropy loss 中集成有softmax
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        loss_record = {'clean': [], 'noise': []}
        loss_list_record = {'clean': [], 'noise': []}
        loss_epoch = [i for i in range(0,10, 1)]
        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_clean_loss = AverageMeter()
            avg_noise_loss = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                true_label, anta_label = data[0:2]
                true_label = true_label.squeeze()
                anta_label = anta_label.squeeze()
                args = data[2:]
                logits = self.parallel_model(*args)
                loss = self.criterion(logits, anta_label)
                elem_loss = self.elem_criterion(logits, anta_label).detach().cpu()
                if epoch in loss_epoch:
                    avg_clean_loss.record(elem_loss[true_label == anta_label].numpy().tolist())
                    avg_noise_loss.record(elem_loss[true_label != anta_label].numpy().tolist())
                avg_clean_loss.update(sum(elem_loss[true_label == anta_label]).item()/sum((true_label == anta_label)).item(), 1)
                avg_noise_loss.update(sum(elem_loss[true_label != anta_label]).item()/sum((true_label != anta_label)).item(), 1)

                score, pred = logits.max(-1)  # (B)
                label_mask = (true_label == anta_label)
                # acc = float((pred == anta_label).long().sum()) / anta_label.size(0)
                acc = float((pred[label_mask] == anta_label[label_mask]).long().sum()) / label_mask.sum()
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            loss_record['clean'].append(avg_clean_loss.avg)
            loss_record['noise'].append(avg_noise_loss.avg)

            if epoch in loss_epoch:
                loss_list_record['clean'].append(avg_clean_loss.list)
                loss_list_record['noise'].append(avg_noise_loss.list)

            # Val
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            self.logger.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                self.logger.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        # json.dump(loss_record, open('../result_visualization/pair_042_loss_result_save.json', 'w'), indent=4)
        json.dump(loss_list_record, open('../result_visualization/symmetric_042_loss_list_save_10.json', 'w'), indent=4)
        self.logger.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        score_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                true_label, anta_label = data[0:2]
                args = data[2:]
                logits = torch.softmax(self.parallel_model(*args), -1)
                score, pred = logits.max(-1)  # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                for i in range(score.size(0)):
                    score_result.append(score[i].item())
                # Log
                label_mask = (true_label == anta_label)
                acc = float((pred == anta_label).long().sum()) / anta_label.size(0)
                # acc = float((pred[label_mask] == anta_label[label_mask]).long().sum()) / label_mask.sum()
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result, score_result)
        self.logger.info('Evaluation result: {}.'.format(result))
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
