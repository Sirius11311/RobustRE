import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder

class LSTMEncoder(BaseEncoder):

    def __init__(self, 
                 token2id,
                 input_size =60, # 50+5+5
                 bidirectional=False,
                 num_layers=2,
                 max_length=128, 
                 hidden_size=256,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 dropout=0,
                 mask_entity=False):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # Hyperparameters
        super(LSTMEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        if bidirectional:
            hidden_size =int(hidden_size/2)
        self.lstm = nn.LSTM(input_size,
                          hidden_size,
                          num_layers,
                          dropout=0.3,
                          bidirectional=bidirectional)

        self.decoder = nn.Sequential(nn.Linear(hidden_size*4, hidden_size*2), nn.Tanh())

    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1), 
                       self.pos2_embedding(pos2)], 2) # (B, L, EMBED)

        x = x.transpose(0, 1) # (L, B, I_EMBED)
        x, (h, c) = self.lstm(x) # (L, B, H_EMBED)
        h = h.transpose(0, 1) # (B, L, I_EMBED)
        h = h.contiguous().view(h.shape[0],-1) # (B, H_EMBED*num_layer)
        # h = self.drop(h)
        h = self.decoder(h)
        # print(h.shape)
        return h

    def tokenize(self, item):
        """
        Args:
            item: input instance, including sentence, entity positions, etc.
        Return:
            index number of tokens and positions
        """
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        # Sentence -> token
        if not is_token:
            if pos_head[0] > pos_tail[0]:
                pos_min, pos_max = [pos_tail, pos_head]
                rev = True
            else:
                pos_min, pos_max = [pos_head, pos_tail]
                rev = False
            sent_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            sent_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            sent_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            if self.mask_entity:
                ent_0 = ['[UNK]']
                ent_1 = ['[UNK]']
            tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
            if rev:
                pos_tail = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_head = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
            else:
                pos_head = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_tail = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
        else:
            tokens = sentence

        # Token -> index
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'],
                                                                  self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id=self.token2id['[UNK]'])

        # Position -> index
        pos1 = []
        pos2 = []
        pos1_in_index = min(pos_head[0], self.max_length)
        pos2_in_index = min(pos_tail[0], self.max_length)
        for i in range(len(tokens)):
            pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
            pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))

        if self.blank_padding:
            while len(pos1) < self.max_length:
                pos1.append(0)
            while len(pos2) < self.max_length:
                pos2.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
            pos1 = pos1[:self.max_length]
            pos2 = pos2[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0)  # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0)  # (1, L)

        return indexed_tokens, pos1, pos2
