import argparse
import logging
import os
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import model.net as net
from model.data_handler import DataHandler
#from evaluate import evaluate


parser.add_argument('--data_dir', default='./data', help='Directory containing datasets')
parser.add_argument('--experiment_dir', default='./experiments/base_model', help='Directory containing the experiment setup')
parser.add_argument('--restore_file', default=None, help='Training checkpoint file name inside experiment_dir (optional)')

if __name__ == '__main__':
    args = parser.parse_args()

    # load parameters from json file
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed for reproduciblity
    torch.manual_seed(230)
    if torch.cuda.is_available(): torch.cuda.manual_seed(230)

    # set logger
    utils.set_logger(os.path.join(args.experiment_dir, 'train.log'))

    # create input data pipeline
    logging.info("Loading the datasets ...")

    # load data and get iterator
    vocab_path = './model/TEXT.Field'
    train_file = os.path.join(args.data_dir, 'train.csv')
    valid_file = os.path.join(args.data_dir, 'val.csv')

    handler = DataHandler()
    handler.load_vocab(vocab_path)
    handler.load_dataset(train_file=train_file, val_file=valid_file)

    train_iter, valid_iter = handler.gen_iterator(params.batch_size)
    train_size, valid_size = handler.data_size

    params.train_size = train_size
    params.valid_size = valid_size
    params.vocab_size = len(handler.vocab.itos)
    params.bme_dim    = len(handler.vocab.itos)*7   # 7 comes from nb=3 + ne=3 + 1

    logging.info("- done.")

    # define model
    model = net.Encoder(params)
    model.to(params.device)

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    model.train()

    for batch in train_iter:
        output = model(batch.input.to(params.device))
        print(output.shape)
