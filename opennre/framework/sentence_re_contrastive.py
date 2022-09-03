from audioop import avg
import os, logging, json, random
import sys
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader, SentenceRELoader_Noise
from .data_loader_contrastive import SentenceRELoaderTrain, SentenceRELoaderKnn
from .utils import AverageMeter, LabelSmoothing
import json
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import shutil
import torch.nn.functional as F
from sklearn.manifold import TSNE



class SentenceREContrastive(nn.Module):

    def __init__(self,
                 model_1,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 logger,
                 batch_size=32,
                 rep_batch_size=700,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 warmup_step=300,
                 opt='sgd',
                 noise_rate=None,
                 noise_mode=None,
                 noise_pattern=None,
                 num_gradual=None,
                 start_epoch = 0,
                 exponent=None,
                 ):

        super().__init__()
        self.max_epoch = max_epoch
        self.logger = logger
        self.noise_mode = noise_mode
        self.noise_pattern = noise_pattern
        self.noise_rate = noise_rate
        self.t1 = -1
        self.t2 = 90
        self.tau_clean = 0.4
        self.max_tau_clean = 0.7
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoaderTrain(
                path = None,
                rel2id = model_1.rel2id,
                tokenizer = model_1.sentence_encoder.tokenize,
                batch_size = batch_size,
                noise_mode = noise_mode,
                noise_pattern = noise_pattern,
                noise_rate = noise_rate
                )

        if val_path != None:
            self.val_loader = SentenceRELoaderKnn(
                path = None,
                rel2id = model_1.rel2id,
                tokenizer = model_1.sentence_encoder.tokenize,
                batch_size = rep_batch_size
                )

        if test_path != None:
            self.test_loader = SentenceRELoaderKnn(
                path = None,
                rel2id = model_1.rel2id,
                tokenizer = model_1.sentence_encoder.tokenize,
                batch_size = rep_batch_size
                )

        # Model
        self.model_1 = model_1
        self.forget_rate = noise_rate
        # self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        loss_weight = False
        if loss_weight:
            self.criterion2 = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion1 = nn.CosineSimilarity(dim=1).cuda()
            self.criterion2 = nn.CrossEntropyLoss().cuda()
            self.criterion3 = nn.CrossEntropyLoss().cuda()
            self.LScriterion = LabelSmoothing(smoothing=0.4)


        self.elem_criterion = nn.CrossEntropyLoss(reduction='none').cuda()  # cross entropy loss 中集成有softmax
        # self.elem_loss_1 = nn.CrossEntropyLoss(reduction='none')  # cross entropy loss 中集成有softmax
        # Params and optimizer
        params_1 = self.model_1.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer_1 = optim.SGD(params_1, lr, weight_decay=weight_decay)
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer_1, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

        # score_reocrd
        self.score_reocrd = []

        # start_epoch
        self.start_epoch = start_epoch

    def kl_div(self, p, q):
        # p, q is in shape (batch_size, n_classes)
        return (p * p.log2() - p * q.log2()).sum(dim=1)

    def js_div(self, p, q):
        # Jensen-Shannon divergence, value is in (0, 1)
        m = 0.5 * (p + q)
        return 0.5 * self.kl_div(p, m) + 0.5 * self.kl_div(q, m)

    def train_model(self, metric='acc'):
        global_step = 0
        best_acc = 0
        loss_record = {'clean': [], 'noise': []}
        loss_list_record = {'clean': [], 'noise': []}
        loss_epoch = [i for i in range(0, 500, 5)]


        for epoch in range(self.max_epoch):
            if epoch > self.max_epoch:
                self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            avg_acc = AverageMeter()
            losses_cr = AverageMeter()
            losses_cf = AverageMeter()
            losses_jcf = AverageMeter()
            avg_clean_loss = AverageMeter()
            avg_noise_loss = AverageMeter()
            slt_clean_count = AverageMeter()
            slt_true_count = AverageMeter()
            total_true_count = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                
                batch_anta_labels, batch_true_labels, batch_pairs1, batch_pairs2 \
                = data[0][0].cuda(), data[0][1].cuda(), data[1][0].cuda(), data[1][1].cuda()  # (batch_size*4*max_length)
                

                p1, p2, z1, z2, c1, c2, jc1, jc2 = self.model_1(
                    [torch.transpose(batch_pairs1, 1, 0).cuda(), torch.transpose(batch_pairs2, 1, 0).cuda()], epoch, self.t1)
                

                N, C = c1.shape
                given_label = torch.full(size=(N, C), fill_value=0.1/(C - 1)).cuda()
                given_label.scatter_(dim=1, index=torch.unsqueeze(batch_anta_labels, dim=1), value=1-0.1)
                probs_1 = F.softmax(c1, dim=1)
                prob_clean = 1 - self.js_div(probs_1, given_label).detach().cpu()
                if epoch < self.t1:
                    threshold_clean = 0
                if epoch > self.t1 and epoch < self.t2:
                    # threshold_clean = min(self.tau_clean * epoch / self.t2, self.tau_clean)
                    threshold_clean = self.tau_clean
                if epoch> self.t1 and epoch >= self.t2:
                    threshold_clean = (self.max_tau_clean - self.tau_clean) * (epoch - self.t2) / (self.max_epoch - self.t2) + self.tau_clean
                clean_index = prob_clean> threshold_clean
                # clean_index = clean_index.squeeze()
                slt_clean_count.update(sum(clean_index).item(), 1)
                select1 = batch_true_labels[clean_index] == batch_anta_labels[clean_index]
                slt_true_count.update(sum(select1), 1)
                total_index = batch_true_labels == batch_anta_labels
                total_true_count.update(sum(total_index).item(), 1)

                loss_cf_elem = (self.elem_criterion(c1, batch_anta_labels) + self.elem_criterion(c2, batch_anta_labels)) * 0.5
                loss_cf = (self.criterion2(c1, batch_anta_labels) + self.criterion2(c2, batch_anta_labels)) * 0.5

                if epoch in loss_epoch:
                    # avg_clean_loss.record(prob_clean[batch_true_labels == batch_anta_labels].numpy().tolist())
                    # avg_noise_loss.record(prob_clean[batch_true_labels != batch_anta_labels].numpy().tolist())
                    avg_clean_loss.record(loss_cf_elem[batch_true_labels == batch_anta_labels].cpu().detach().numpy().tolist())
                    avg_noise_loss.record(loss_cf_elem[batch_true_labels != batch_anta_labels].cpu().detach().numpy().tolist())

                if epoch <= self.t1:
                    loss =  loss_cf
                    losses_cf.update(loss_cf.cpu().item(), 1)

                else:
                    aa1 = min(0.95, epoch/(0.001+ self.max_epoch))
                    loss_cr = -(self.criterion1(p1, z2).mean() + self.criterion1(p2, z1).mean()) * 0.5
                    loss_jcf = (self.LScriterion(jc1, batch_anta_labels) + \
                        self.LScriterion(jc2, batch_anta_labels)) * 0.5
                    # loss_jcf = (self.LScriterion(jc1[clean_index], batch_anta_labels[clean_index]) + \
                    #     self.LScriterion(jc2[clean_index], batch_anta_labels[clean_index])) * 0.5
                    loss = (1-aa1)*loss_cr + loss_cf + aa1*loss_jcf
                    # loss = loss_cf + loss_jcf
                    # # loss = loss_jcf
                    # # loss = loss_cr + loss_cf + loss_jcf


                    losses_cr.update(loss_cr.cpu().item(), 1)
                    losses_cf.update(loss_cf.cpu().item(), 1)
                    losses_jcf.update(loss_jcf.cpu().item(), 1)

                score, pred = (c1+c2).max(-1) # (B)
                acc = float((pred == batch_anta_labels).long().sum()) / batch_true_labels.size(0)
                avg_acc.update(acc, 1)
                t.set_postfix(avg_acc = avg_acc.avg,loss_cr=losses_cr.avg, loss_cf=losses_cf.avg, loss_jcf=losses_jcf.avg, epoch=epoch)
                # Optimize
                loss.backward()
                self.optimizer_1.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer_1.zero_grad()

                global_step += 1

            # if epoch in loss_epoch:
            #     loss_list_record['clean'].append(avg_clean_loss.list)
            #     loss_list_record['noise'].append(avg_noise_loss.list)

            if epoch in loss_epoch:
                loss_list_record['clean'].append([np.mean(avg_clean_loss.list), np.std(avg_clean_loss.list)])
                loss_list_record['noise'].append([np.mean(avg_noise_loss.list), np.std(avg_noise_loss.list)])

            slt_p = slt_true_count.sum / (slt_clean_count.sum + 1e-10)
            slt_r = slt_true_count.sum / (total_true_count.sum + 1e-10)
            slt_f1 = 2*slt_p*slt_r /(slt_p + slt_r)
            self.logger.info(f'clean select metric: p {slt_p} | r {slt_r} | f1 {slt_f1}')


            # Val
            if epoch % 1 ==0:
                self.logger.info("=== Epoch %d val ===" % epoch)
                avg_acc = self.eval_model(self.val_loader, epoch)
                self.logger.info('Metric {} current / best: {} / {}'.format(metric, avg_acc, best_acc))     
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    self.logger.info("Best performance !!!")
                    torch.save({'state_dict': self.model_1.state_dict()}, f'ckpt/{self.ckpt}.pth.tar')
                # self.save_checkpoint({
                #     'epoch': epoch + 1,
                #     'arch': 'pcnn',
                #     'state_dict': self.model_1.state_dict(),
                #     'optimizer': self.optimizer_1.state_dict(),
                # }, is_best=False, filename=f'../ckpt/wiki40/pcnn/latest/{self.ckpt}.pth.tar')
            # self.logger.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            # if result[metric] > best_metric:
            #     self.logger.info("Best ckpt and saved.")
            #     folder_path = '/'.join(self.ckpt.split('/')[:-1])
            #     if not os.path.exists(folder_path):
            #         os.mkdir(folder_path)
            #     torch.save({'state_dict': self.model_1.state_dict()}, self.ckpt)
            #     best_metric = result[metric]
        # json.dump(loss_record, open(f'../result_visualization/loss_record/{self.noise_mode}/{self.noise_pattern}/pcnn_{self.noise_rate}.json', 'w'), indent=4)
        # json.dump(loss_list_record, open('../result_visualization/finetune_sphere_openset_symmetric_080_loss_1.json', 'w'), indent=4)
        # self.logger.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader, epoch):
        self.eval()
        knn_batch_size = 700
        kn = 5
        # rep_record = np.zeros(shape=(5600, 690))  #
        # label_record = np.zeros(shape=(5600,))
        pred_result = []

        avg_acc = AverageMeter()
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                input_data = data[1]  # TODO 这里还需要修改
                label = data[2].cuda()
                h, logits = self.model_1.get_rep(torch.transpose(input_data, 1, 0), epoch, self.t1)
                score, pred = logits.max(-1) # (B)
                # rep_record[iter * knn_batch_size:(iter + 1) * knn_batch_size] = h.cpu().numpy()[:]
                # label_record[iter * knn_batch_size:(iter + 1) * knn_batch_size] = label.cpu().numpy()[:]

                    # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        return avg_acc.avg
            # result = eval_loader.dataset.eval(pred_result)
            # logger.info('Evaluation result: {}.'.format(result))


        # knn = KNeighborsClassifier(n_neighbors=kn, metric='cosine')
        # scores = cross_val_score(knn, rep_record, label_record, cv=6, scoring='accuracy')
        # self.score_reocrd.append(1 - scores.mean())
        # self.logger.info(self.score_reocrd)
        # np.savetxt(f'../save_data/wiki40/contr_pcnn/{self.ckpt}_se{self.start_epoch}.txt', self.score_reocrd, fmt='%.4f')

        # if self.score_reocrd[-1] <= np.min(self.score_reocrd):
        #     self.save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': 'pcnn',
        #         'state_dict': self.model_1.state_dict(),
        #         'optimizer': self.optimizer_1.state_dict(),
        #     }, is_best=False, filename=f'../ckpt/wiki40/pcnn/best/{self.ckpt}.pth.tar')


    def get_rep(self, eval_loader):
        self.eval()
        count = 0
        knn_batch_size = 200
        if self.noise_mode == 'closeset':
            rep_record = np.zeros(shape=(25200, 690))  #
            label_record = np.zeros(shape=(25200,))
        if self.noise_mode == 'openset':
            rep_record = np.zeros(shape=(25200, 690))  #
            label_record = np.zeros(shape=(25200,))            
        # with torch.no_grad():
        #     for iter, data in enumerate(eval_loader):
        #         if torch.cuda.is_available():
        #             for i in range(len(data)):
        #                 try:
        #                     data[i] = data[i].cuda()
        #                 except:
        #                     pass
        #         input_data = data[1]  
        #         label = data[2].cuda()
        #         h, _ = self.model_1.get_rep(torch.transpose(input_data, 1, 0), 0, -1)
        #         rep_record[iter * knn_batch_size:(iter + 1) * knn_batch_size] = h.cpu().numpy()[:]
        #         label_record[iter * knn_batch_size:(iter + 1) * knn_batch_size] = label.cpu().numpy()[:]

        with torch.no_grad():
            for iter, data in enumerate(eval_loader):
                batch_anta_labels, batch_true_labels, batch_pairs1, batch_pairs2 \
                = data[0][0].cuda(), data[0][1].cuda(), data[1][0].cuda(), data[1][1].cuda()  # (batch_size*4*max_length)
                h = self.model_1.get_rep_openset(
                    [torch.transpose(batch_pairs1, 1, 0).cuda(), torch.transpose(batch_pairs2, 1, 0).cuda()])
                rep_record[iter * knn_batch_size:(iter + 1) * knn_batch_size] = h.cpu().numpy()[:]
                label_record[iter * knn_batch_size:(iter + 1) * knn_batch_size] = batch_true_labels.cpu().numpy()[:]

                count = count + knn_batch_size
                if  count >= 25200:
                    break

        # data_index = random.sample([i for i in range(25240)], 2000)
        if self.noise_mode == 'closeset':
            data_index = label_record< 40
        if self.noise_mode == 'openset':
            data_index = label_record< 80
        
        rep_record = rep_record[data_index]
        label_record = label_record[data_index]

        # if self.noise_mode == 'openset':
        #     data_index = label_record <10
        #     open_index = label_record >= 40
        #     open_num = sum(open_index)
        #     close_record = rep_record[data_index]
        #     close_label = label_record[data_index]
            
        #     open_record = rep_record[open_index]
        #     open_label = label_record[open_index]

        #     open_select = random.sample([i for i in range(open_num)], 5000)
        #     select_open_record = [open_record[i] for i in open_select]
        #     select_open_label = [open_label[i] for i in open_select]
            

            # rep_record = np.concatenate([close_record, select_open_record], axis=0) 
            # label_record = np.concatenate([close_label, select_open_label], axis=0)

        self.logger.info('TSNE start!')
        tsne = TSNE(n_components=2, perplexity=30, learning_rate= 300.0)
        tsne.fit_transform(rep_record)
        np.save(f'../save_rep/UnsCL_all_{self.noise_mode}_new_embedding_all_30_300.npy', np.array(tsne.embedding_))
        np.save(f'../save_rep/UnsCL_all_{self.noise_mode}_new_label_all_30_300.npy', label_record)
        self.logger.info('TSNE finish!')

    def load_state_dict(self, state_dict):
        self.model_1.load_state_dict(state_dict)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
