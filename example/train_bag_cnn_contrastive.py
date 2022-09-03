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
import copy


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
parser.add_argument('--metric', default='acc', choices=['micro_f1', 'auc', 'max_micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='noise_wiki40',
                    choices=['none', 'wiki_distant', 'nyt10', 'nyt10m', 'wiki20m', 'clean_wiki10', 'noise_wiki10',
                             'noise_wiki80', 'noise_wiki40', 'nyt24', 'nyt24_noise_control', 'noise_wiki40_contrastive'],
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
parser.add_argument('--noise_mode', default='openset', choices=['openset', 'None', 'closeset'],
                    help='Noise mode of data')
parser.add_argument('--noise_pattern', default='symmetric', choices=['symmetric', 'pair'],
                    help='Noise mode of data')
parser.add_argument('--noise_rate', default=0.45, type=float,
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
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Weight decay')
parser.add_argument('--max_length', default=132, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=1000, type=int,
                    help='Max number of training epochs')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Max number of training epochs')
# Others
parser.add_argument('--seed', default=1, type=int,
                    help='Random seed')
parser.add_argument('--log_file', default='', type=str,
                    help='file of log')

# Exp
parser.add_argument('--encoder', default='cnn', choices=['pcnn', 'cnn', 'lstm'])
parser.add_argument('--aggr', default='att', choices=['one', 'att', 'avg'])
parser.add_argument('--pred', default='softmax', choices=['softmax', 'sigmoid'])


# Co-teaching
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=0.5,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')



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
ckpt = args.ckpt

if args.dataset == 'nyt24' or args.dataset == 'nyt24_noise_control'  or args.dataset == 'nyt24_2.0':
    # opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_train_reduce.txt'.format(args.dataset))
    # args.train_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    args.val_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_dev.txt'.format(args.dataset))
    if not os.path.exists(args.val_file):
        logger.info("Cannot find the validation file. Use the test file instead.")
        args.val_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    args.test_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    args.rel2id_file = os.path.join(benchmark_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
elif args.dataset == 'noise_wiki40':
    # opennre.download(args.dataset, root_path=root_path)
    # args.train_file = os.path.join(benchmark_path, 'benchmark', args.dataset, 'wiki80_raw_system_train.txt')
    args.train_file = os.path.join(benchmark_path, 'benchmark', args.dataset, 'wiki_c40_n40_045_balance.txt')
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
opennre.download('glove', root_path=root_path)
word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.PCNNEncoder(
    token2id=word2id,
    max_length=args.max_length,
    word_size=50,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.0
)

# Define the model
model = opennre.model.SoftmaxNNContrastive(sentence_encoder, len(rel2id)//2, rel2id)


# Define the whole training framework (sentence level)
framework = opennre.framework.SentenceREContrastive(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model_1=model,
    ckpt=ckpt,
    logger=logger,
    batch_size=args.batch_size,
    rep_batch_size = 630,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt=args.optim,
    noise_rate=args.noise_rate,
    noise_mode=args.noise_mode,
    noise_pattern=args.noise_pattern,
    num_gradual=args.num_gradual,
    exponent=args.exponent,
    start_epoch = args.start_epoch
)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# framework.get_rep(framework.test_loader)

# Test the model
# framework.load_state_dict(torch.load(ckpt)['state_dict'])
# framework.get_rep(framework.train_loader)

# # Print the result
# logger.info('Test set results:')
# logger.info(result)
