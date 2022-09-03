import torch
from torch import nn, optim
from .base_model import SentenceRE
import sys
import torch.nn.functional as F

class SoftmaxNNContrastive(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.drop_noise = nn.Dropout(0.01)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel



        dim = self.sentence_encoder.hidden_size
        prev_dim = self.sentence_encoder.hidden_size
        pred_dim = 256
        self.fc1 = nn.Linear(prev_dim, dim)
        self.fc2 = nn.Linear(self.sentence_encoder.hidden_size, 20)
        self.classifier = nn.Linear(self.sentence_encoder.hidden_size, num_class, bias=False)
        self.classifier2 = nn.Linear(20, num_class, bias=False)


        # build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True),  # first layer
                                       nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True),  # second layer
                                       self.fc1,
                                       nn.BatchNorm1d(dim, affine=False))  # output layer
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer


    def forward(self, x, epoch, t1):
        x1, x2 = x


        h1 = self.sentence_encoder(*x1)
        h2 = self.sentence_encoder(*x2)  # raw sentence

        z1 = self.projector(h1)  # NxC
        z2 = self.projector(h2)  # NxC

        p1 = self.predictor(z1)  # CxC
        p2 = self.predictor(z2)  # CxC

        # h1 = self.drop(h1)
        # h2 = self.drop(h2)

        # norm1 = torch.norm(h1, dim=1)
        # norm1 = torch.norm(h1, dim=1)
        # h1 =h1.detach()
        # h2 =h2.detach()

        # h1 = F.normalize(h1, dim=1)
        # h2 = F.normalize(h2, dim=1)


        c1 = self.classifier(h1.detach())  # NxC
        c2 = self.classifier(h2.detach())  # NxC
        # c1 = self.classifier(h1*5)  # NxC
        # c2 = self.classifier(h2*5)  # NxC

        if epoch > t1:
            h11 = self.fc2(h1)
            h22 = self.fc2(h2)
            jc1 = self.classifier2(h11)
            jc2 = self.classifier2(h22)
            return p1, p2, z1.detach(), z2.detach(), c1, c2, jc1, jc2
        else:
            return p1, p2, z1.detach(), z2.detach(), c1, c2, [], []


    def get_rep_openset(self, x):
        x1, x2 = x


        h2 = self.sentence_encoder(*x2)  # raw sentence
        h22 = self.fc2(h2)  

        return h22


    def get_rep(self, x, epoch, t1):

        h1 = self.sentence_encoder(*x)

        if epoch <= t1:
            c1 = self.classifier(h1)
            return h1, c1

        else:
            # z1 = self.projector(h1)  # NxC
            h11 = self.fc2(h1)
            jc1 = self.classifier2(h11)
            return h11, jc1


    def logit_to_score(self, logits):
        return torch.softmax(logits, -1)
