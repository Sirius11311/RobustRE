import torch
import torch.utils.data as data
import random, json, logging
from tqdm import tqdm
import sys
import operator

class SentenceREDataset_ContrastiveTrain(data.Dataset):
    def __init__(self, path, rel2id, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.path = '../benchmark/noise_wiki40/wiki80_raw_system_train.txt'
        self.Y = []

        # f = open(self.path)
        # self.data = []
        # for line in f.readlines():
        #     line = line.rstrip()
        #     if len(line) > 0:
        #         line = eval(line)
        #         self.data.append(line)
        #         self.Y.append(rel2id[line['true_label']])
        # f.close()
        # logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data),
        #                                                                                     len(self.rel2id)))

        # self.Y = torch.tensor(self.Y)
        # self.true_label = self.Y
        # self.anta_label = self.Y

        # data1, _ = self.get_data('../benchmark/noise_wiki40_contrastive/wiki_c40_n40_045_balance_aug1.txt')
        # data2, _ = self.get_data('../benchmark/noise_wiki40_contrastive/wiki_c40_n40_045_balance_aug2.txt')
        # data3, Y = self.get_data('../benchmark/noise_wiki40_contrastive/wiki_c40_n40_045_balance_aug3.txt')
        # data4 = self.get_data('data/nyt24/nyt24_train_part.txt')
        # data4 = self.get_data('../benchmark/noise_wiki40/wiki_c40_n40_045_balance.txt')
        data4, _ = self.get_data('../benchmark/noise_wiki40/wiki_c40_n40_045_balance.txt')

        # self.pairs1, self.pairs2, self.anta_label, self.true_label = self.create_test_pairs([data1, data2, data3, data4])
        self.pairs1, self.pairs2, self.anta_label, self.true_label = self.create_test_pairs([data4, data4])


        # self.pairs1, self.anta_label, self.true_label = self.create_test_pairs([data4])

        # print(operator.eq(self.anta_label,  self.true_label))
        # sys.exit()

        # self.anta_label = Y
        # self.true_label = Y


        # self.pairs1, self.pairs2, self.anta_label, self.true_label = self.create_test_pairs([data1, data2, data3, data4])

    def __len__(self):
        return len(self.pairs1)

    def __getitem__(self, item):
        # seq1 = list(self.tokenizer(self.pairs1[item]))
        # seq2 = list(self.tokenizer(self.pairs2[item]))
        return [self.pairs1[item], self.pairs2[item], self.anta_label[item], self.true_label[item]]
        # seq1 = list(self.tokenizer(self.pairs1[item]))
        # return [seq1, self.anta_label[item], self.true_label[item]]

    def get_data(self, path):
        f = open(path)
        data = []
        label = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                line = eval(line)
                data.append(line)
                label.append(self.rel2id[line['true_label']])
        f.close()
        logging.info(
            "Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(data),
                                                                                   len(self.rel2id)))
        return data, label

    def create_test_pairs(self, data_list):
        random.seed(0)
        pairs1 = []
        pairs2 = []
        anta_label = []
        true_label = []

        for i in tqdm(range(len(data_list[0]))):
            for j in range(len(data_list)-1):
                for k in range(len(data_list)-1, len(data_list)):
                    z1, z2 = data_list[j][i], data_list[k][i]
                    seq1 = list(self.tokenizer(z1))
                    seq2 = list(self.tokenizer(z2))
                    pairs1 += [seq1]
                    pairs2 += [seq2]
                    anta_label.append(self.rel2id[z2['anta_label']])
                    true_label.append(self.rel2id[z2['true_label']])

        return pairs1, pairs2, anta_label, true_label



    def collate_fn(data):
        batch_pairs1 = []
        batch_pairs2 = []
        batch_anta_labels = []
        batch_true_labels = []
        for i in data:
            batch_pairs1.append(torch.cat(i[0], 0))
            batch_pairs2.append(torch.cat(i[1], 0))
            batch_anta_labels.append(i[2])
            batch_true_labels.append(i[3])


        batch_pairs1 = torch.stack(batch_pairs1)
        batch_pairs2 = torch.stack(batch_pairs2)

        batch_true_labels = torch.tensor(batch_true_labels).squeeze().long()  # (B)
        batch_anta_labels = torch.tensor(batch_anta_labels).squeeze().long()  # (B)


        return [batch_anta_labels, batch_true_labels], [batch_pairs1, batch_pairs2]


def SentenceRELoaderTrain(path, rel2id, tokenizer, batch_size,
                          num_workers=8, collate_fn=SentenceREDataset_ContrastiveTrain.collate_fn):
    dataset = SentenceREDataset_ContrastiveTrain(path=path, rel2id=rel2id, tokenizer=tokenizer)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader



class SentenceREDataset_TrainRep(data.Dataset):
    def __init__(self, path, rel2id, tokenizer):  # noise_mode =[pair, symmetric]
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.Y = []

        # Load the file
        path = '../benchmark/noise_wiki40_contrastive/wiki80_raw_test.txt'
        # if path == None:
        #     # path = '../benchmark/noise_wiki40/wiki_c40_n40_045_balance.txt'
        #     # path = 'data/nyt24/nyt24_test.txt'
        #     path = 'data/noise_wiki40/wiki80_raw_test.txt'
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                line = eval(line)
                if rel2id[line['true_label']] < 40:
                    self.data.append(line)
                    self.Y.append(rel2id[line['true_label']])
        f.close()
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data),
                                                                                            len(self.rel2id)))

        self.Y = torch.tensor(self.Y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item))
        return [self.rel2id[item['true_label']],
                self.rel2id[item['anta_label']]] + seq, self.Y[index]  # true_label, anta_label, seq1, seq2, ...

    def collate_fn(data):
        batch_true_labels = []
        batch_anta_labels = []
        batch_seqs = []
        batch_labels = []
        for i in data:
            batch_true_labels.append(i[0][0:1])
            batch_anta_labels.append(i[0][1:2])
            batch_seqs.append(torch.cat(i[0][2:], 0))
            batch_labels.append(i[1])
        batch_true_labels = torch.tensor(batch_true_labels).squeeze().long()  # (B)
        batch_anta_labels = torch.tensor(batch_anta_labels).squeeze().long()  # (B)
        batch_labels = torch.tensor(batch_labels).long()  # (B)
        batch_seqs = torch.stack(batch_seqs)


        return [batch_true_labels, batch_anta_labels], batch_seqs, batch_labels


def SentenceRELoaderKnn(path, rel2id, tokenizer, batch_size, num_workers=8,
                        collate_fn=SentenceREDataset_TrainRep.collate_fn):
    dataset = SentenceREDataset_TrainRep(path=path, rel2id=rel2id, tokenizer=tokenizer)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader







if __name__ == '__main__':
    rel2id = json.load(open('data/noise_wiki40/wiki80_rel2id.json'))

