import argparse
import logging
import os
import tqdm
from random import randrange

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import model.net as net
from model.data_handler import DataHandler


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='Directory containing datasets')
parser.add_argument('--experiment_dir', default='./experiments/base_model', help='Directory containing the experiment setup')
parser.add_argument('--restore_file', default=None, help='Training checkpoint file name inside experiment_dir (optional)')


def train(model, optimizer, train_iterator, params):
    """Train the model on one epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        train_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    total_loss = 0.
    num_steps = params.train_size//params.batch_size

    for batch in tqdm.tqdm(train_iterator, total=num_steps):
        loss = torch.tensor(0).float()
        output = model(batch.input.to(params.device).float())

        # compute loss to minimize distance between center word and
        # context words based on params.context_window size
        for i in range(1, params.context_window+1):
            loss += net.cos_embedding_loss(output[i:, :, :], output[:-i, :, :])

        # negative-samples loss
        # negative-samples are taken by randomly rolling batch
        for i in range(params.negative_samples):
            loss += net.cos_embedding_loss(output, torch.roll(output, randrange(params.batch_size), 1), True)

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info("- Average training loss: {}".format(total_loss/num_steps))


def evaluate(model, optimizer, val_iterator, params):
    """Evaluate the model on evaluation data

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        val_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """
    with torch.no_grad():
        model.eval()

        total_loss = 0.
        num_steps = params.val_size//params.batch_size

        for batch in tqdm.tqdm(val_iterator, total=num_steps):
            output = model(batch.input.to(params.device).float())

            # compute loss to minimize distance between center word and
            # context words based on params.context_window size
            for i in range(1, params.context_window+1):
                total_loss += net.cos_embedding_loss(output[i:, :, :], output[:-i, :, :]).item()

            # negative-samples loss
            # negative-samples are taken by randomly rolling batch
            for i in range(params.negative_samples):
                total_loss += net.cos_embedding_loss(output, torch.roll(output, randrange(params.batch_size), 1), True).item()

        logging.info("- Average evaluation loss: {}".format(total_loss/num_steps))

    return {'loss': total_loss/num_epochs}



def train_and_evaluate(model, optimizer, train_iterator, val_iterator, params):
    """Train the model and evaluate every epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        train_iterator: (generator) a generator that generates batches of data and labels
        val_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """

    # reload weights from checkpoint if specified
    if args.restore_file:
        restore_path = os.path.join(args.experiment_dir, args.restore_file)

        logging.info("Restoring parameters from {}".format(restore_path))
        if not os.path.exists(restore_path):
            raise ("File doesn't exist")

        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_val_loss = float("Inf")

    # training on num_epochs
    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epochs+1, params.num_epochs))

        # run one epoch
        train(model, optimizer, train_iterator, params)
        val_metric = evaluate(model, optimizer, val_iterator, params)

        # save best model regards on loss
        if val_metric['loss'] <= best_val_loss:
            best_val_loss = val_metric['loss']

            path = os.path.join(args.experiment_dir, 'best_loss.pth.tar')
            torch.save({'epoch': epoch+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': val_metric['loss']},
                        path)

        # save latest model
        path = os.path.join(args.experiment_dir, 'latest.pth.tar')
        torch.save({'epoch': epoch+1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': val_metric['loss']},
                    path)


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
    params.bme_dim    = params.vocab_size*7   # 7 comes from nb=3 + ne=3 + 1

    logging.info("- done.")

    # define model
    model = net.Encoder(params)
    model.to(params.device)

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    