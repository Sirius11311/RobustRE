# coding:utf-8
import sys, json

# print(sys.path)
sys.path.append('..')
import torch
import os
import numpy as np
import opennre
import argparse
import logging
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(path):
    logger = logging.getLogger('nhy')
    logger.setLevel(logging.INFO)
    fmt1 = logging.Formatter(fmt='%(name)s | %(asctime)s | %(message)s', datefmt='%m/%d/%Y %H:%M:%S ')
    fmt2 = logging.Formatter(fmt='%(name)s | %(asctime)s | %(message)s', datefmt='%m/%d/%Y %H:%M:%S ')
    hander_sc = logging.StreamHandler()
    hander_fl = logging.FileHandler(path)
    hander_sc.setLevel(logging.INFO)
    hander_fl.setLevel(logging.INFO)
    hander_sc.setFormatter(fmt1)
    hander_fl.setFormatter(fmt2)
    logger.addHandler(hander_fl)
    # logger.addHandler(hander_sc)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='',
                    help='Checkpoint name')
parser.add_argument('--result', default='',
                    help='Save result name')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')

# Data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='noise_wiki40',
                    choices=['none', 'wiki_distant', 'nyt10', 'nyt10m', 'wiki20m', 'clean_wiki10', 'noise_wiki10',
                             'noise_wiki80', 'noise_wiki40'],
                    help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
                    help='Training data file')
parser.add_argument('--val_file', default='', type=str,
                    help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
                    help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
                    help='Relation to ID file')

# noise

parser.add_argument('--noise_mode', default='symmetric', choices=['symmetric', 'pair'],
                    help='Noise mode of data')
parser.add_argument('--noise_rate', default=0.3, type=float,
                    help='Noise rate of data')

# Data level
parser.add_argument('--data_level', default='sentence', choices=['sentence', 'bag'],
                    help='data level of the task')

# Bag related
parser.add_argument('--bag_size', type=int, default=0,
                    help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=160, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=0.5, type=float,
                    help='Learning rate')
parser.add_argument('--optim', default='sgd', type=str,
                    help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Weight decay')
parser.add_argument('--max_length', default=128, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=100, type=int,
                    help='Max number of training epochs')

# Others
parser.add_argument('--seed', default=42, type=int,
                    help='Random seed')
parser.add_argument('--log_file', default='', type=str,
                    help='file of log')

# Exp
parser.add_argument('--encoder', default='cnn', choices=['pcnn', 'cnn', 'lstm'])
parser.add_argument('--aggr', default='att', choices=['one', 'att', 'avg'])
parser.add_argument('--pred', default='softmax', choices=['softmax', 'sigmoid'])

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

# Set logger
logger = set_logger(args.log_file)

# Some basic settings
root_path = '..'
benchmark_path = '..'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}'.format(args.dataset, 'pcnn_att')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

# if args.dataset != 'none':
#     # opennre.download(args.dataset, root_path=root_path)
#     args.train_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
#     args.val_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
#     if not os.path.exists(args.val_file):
#         logger.info("Cannot find the validation file. Use the test file instead.")
#         args.val_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
#     args.test_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
#     args.rel2id_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
if args.dataset != 'none':
    # opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(benchmark_path, 'benchmark', args.dataset, 'wiki80_raw_system_train.txt')
    args.val_file = os.path.join(benchmark_path, 'benchmark', args.dataset, 'wiki80_val.txt')
    if not os.path.exists(args.val_file):
        logger.info("Cannot find the validation file. Use the test file instead.")
        args.val_file = os.path.join(benchmark_path, 'benchmark', args.dataset, 'wiki80_raw_test.txt')
    args.test_file = os.path.join(benchmark_path, 'benchmark', args.dataset, 'wiki80_raw_test.txt')
    args.rel2id_file = os.path.join(benchmark_path, 'benchmark', args.dataset, 'wiki80_rel2id.json')
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(
            args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception(
            '--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logger.info('Arguments:')
for arg in vars(args):
    logger.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))

# Download glove
# opennre.download('glove', root_path=root_path)
word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
if args.encoder == 'pcnn':
    sentence_encoder_1 = opennre.encoder.PCNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )
    sentence_encoder_2 = opennre.encoder.PCNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )
elif args.encoder == 'cnn':
    sentence_encoder_1 = opennre.encoder.CNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )
    sentence_encoder_2 = opennre.encoder.CNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )
elif args.encoder == 'lstm':
    sentence_encoder_1 = opennre.encoder.LSTMEncoder(
        token2id=word2id,
        max_length=args.max_length,
        bidirectional=True,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        word2vec=word2vec,
        dropout=0.5
    )
    sentence_encoder_2 = opennre.encoder.LSTMEncoder(
        token2id=word2id,
        max_length=args.max_length,
        bidirectional=True,
        word_size=50,
        position_size=5,
        hidden_size=230,
        blank_padding=True,
        word2vec=word2vec,
        dropout=0.5
    )
else:
    raise NotImplementedError

# Define the model
if args.data_level == 'bag':
    if args.aggr == 'att':
        model_1 = opennre.model.BagAttention(sentence_encoder_1, len(rel2id), rel2id)
        model_2 = opennre.model.BagAttention(sentence_encoder_2, len(rel2id), rel2id)
    elif args.aggr == 'avg':
        model_1 = opennre.model.BagAverage(sentence_encoder_1, len(rel2id), rel2id)
        model_2 = opennre.model.BagAverage(sentence_encoder_2, len(rel2id), rel2id)
    elif args.aggr == 'one':
        model_1 = opennre.model.BagOne(sentence_encoder_1, len(rel2id), rel2id)
        model_2 = opennre.model.BagOne(sentence_encoder_2, len(rel2id), rel2id)
    else:
        raise NotImplementedError
if args.data_level == 'sentence':
    if args.pred == 'softmax':
        model_1 = opennre.model.SoftmaxNN(sentence_encoder_1, 40, rel2id)
        model_2 = opennre.model.SoftmaxNN(sentence_encoder_2, 40, rel2id)
    elif args.aggr == 'sigmoid':
        model_1 = opennre.model.SigmoidNN(sentence_encoder_1, 40, rel2id)
        model_2 = opennre.model.SigmoidNN(sentence_encoder_2, 40, rel2id)

# # Define the whole training framework (bag level)
# framework = opennre.framework.BagRE(
#     train_path=args.train_file,
#     val_path=args.val_file,
#     test_path=args.test_file,
#     model=model,
#     ckpt=ckpt,
#     batch_size=args.batch_size,
#     max_epoch=args.max_epoch,
#     lr=args.lr,
#     weight_decay=args.weight_decay,
#     opt=args.optim,
#     bag_size=args.bag_size)

# Define the whole training framework (sentence level)
framework = opennre.framework.SentenceRENoise(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model_1=model_1,
    model_2=model_2,
    ckpt=ckpt,
    logger=logger,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt=args.optim,
    noise_rate=args.noise_rate,
    noise_mode=args.noise_mode
)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
logger.info('Test set results:')
logger.info(result)
# logger.info('AUC: %.5f' % (result['auc']))
# logger.info('Maximum micro F1: %.5f' % (result['max_micro_f1']))
# logger.info('Maximum macro F1: %.5f' % (result['max_macro_f1']))
# logger.info('Micro F1: %.5f' % (result['micro_f1']))
# logger.info('Macro F1: %.5f' % (result['macro_f1']))
# logger.info('P@100: %.5f' % (result['p@100']))
# logger.info('P@200: %.5f' % (result['p@200']))
# logger.info('P@300: %.5f' % (result['p@300']))

# # Save precision/recall points
# np.save('result/{}_p.npy'.format(args.result), result['np_prec'])
# np.save('result/{}_r.npy'.format(args.result), result['np_rec'])
# json.dump(result['max_micro_f1_each_relation'], open('result/{}_mmicrof1_rel.json'.format(args.result), 'w'), ensure_ascii=False)
