import os
import argparse

import torchtext.data as data
from model.data_handler import DataHandler, bme


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='./data/train.csv', help='Training dataset in csv to build vocabulary upon')
parser.add_argument('--max_vocab_size', default=20000, help='Maximum vocabulary size to be built')

if __name__ == '__main__':
	args = parser.parse_args()

	print("Loading dataset ...")
	data_handler = DataHandler(train_file=args.train_file)
	print("- done.")

	nesting_field = data.Field(tokenize=list, postprocessing=bme)
	CHARS = data.NestedField(nesting_field)

	print("Building word vocabulary ...")
    data_handler.build_vocab(field=CHARS,
                             target_path='./model/TEXT.Field',
                             max_size=args.max_vocab_size)
    print("- done.")