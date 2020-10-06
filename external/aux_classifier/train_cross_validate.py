# coding: utf-8

import argparse
import codecs
import dill as pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from itertools import product as p
from torch.utils.serialization import load_lua
from tqdm import tqdm, tqdm_notebook, tnrange

# Import lib
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import aux_classifier.utils as utils
import aux_classifier.representations as repr
import aux_classifier.data_loader as data_loader

def main():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--train-source', dest='train_source', required=True,
                    help='Location of train source file')
    parser.add_argument('--train-aux-source', dest='train_aux_source',
                    help='Location of aux train source file (BPE/CHAR)')
    parser.add_argument('--train-labels', dest='train_labels', required=True,
                    help='Location of train source labels')
    parser.add_argument('--train-activations', dest='train_activations', required=True,
                    help='Location of train source activations')
    
    parser.add_argument('--exp-type', dest='exp_type', 
                    choices=['word', 'charcnn', 'bpe_avg', 'bpe_last', 'char_avg', 'char_last'],
                    default='word', required=True,
                    help='Type of experiment')

    parser.add_argument('--task-specific-tag', dest='task_specific_tag', 
                    required=True, help='Tag incase test has unknown tags')

    parser.add_argument('--max-sent-l', dest='max_sent_l', type=int,
                    default=250, help='Tag incase test has unknown tags')

    parser.add_argument('--num-folds', dest='n_folds', type=int,
                    default=5, help='Number of folds for cross validation')

    parser.add_argument('--output-dir', dest='output_dir', 
                    required=True, help='Location to save all results')

    args = parser.parse_args()

    print("Creating output directory...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Constants
    NUM_EPOCHS = 10
    BATCH_SIZE = 512
    BRNN = 2

    print("Loading activations...")
    train_activations = data_loader.load_activations(args.train_activations)
    print("Number of train sentences: %d"%(len(train_activations)))

    if args.exp_type == 'word' or args.exp_type == 'charcnn':
        train_tokens = data_loader.load_data(args.train_source, args.train_labels, train_activations, args.max_sent_l)
    else:
        train_tokens = data_loader.load_aux_data(args.train_source, args.train_labels, args.train_aux_source, train_activations, args.max_sent_l)

    NUM_TOKENS = sum([len(t) for t in train_tokens['target']])
    print('Number of total train tokens: %d'%(NUM_TOKENS))

    if args.exp_type != 'word' and args.exp_type != 'charcnn':
        NUM_SOURCE_AUX_TOKENS = sum([len(t) for t in train_tokens['source_aux']])
        print('Number of AUX source words: %d'%(NUM_SOURCE_AUX_TOKENS)) 

    NUM_SOURCE_TOKENS = sum([len(t) for t in train_tokens['source']])
    print('Number of source words: %d'%(NUM_SOURCE_TOKENS)) 

    NUM_NEURONS = train_activations[0].shape[1]
    print('Number of neurons: %d'%(NUM_NEURONS))

    if args.exp_type == 'bpe_avg':
        train_activations = repr.bpe_get_avg_activations(train_tokens, train_activations)
    elif args.exp_type == 'bpe_last':
        train_activations = repr.bpe_get_last_activations(train_tokens, train_activations, is_brnn=(BRNN == 2))
    elif args.exp_type == 'char_avg':
        train_activations = repr.char_get_avg_activations(train_tokens, train_activations)
    elif args.exp_type == 'char_last':
        train_activations = repr.char_get_last_activations(train_tokens, train_activations, is_brnn=(BRNN == 2))

    print("Creating train tensors...")
    X, y, mappings = utils.create_tensors(train_tokens, train_activations, args.task_specific_tag)

    label2idx, idx2label, src2idx, idx2src = mappings

    num_samples = X.shape[0]
    num_samples_per_fold = num_samples/args.n_folds
    shuffle_idx = np.random.permutation(np.arange(num_samples))

    overall_train_accuracies = []
    overall_test_accuracies = []

    for fold in range(args.n_folds):
        start_idx = int(num_samples_per_fold * fold)
        end_idx = min(int(num_samples_per_fold * (fold+1)), num_samples)

        train_mask = np.ones(num_samples, np.bool)
        train_mask[start_idx:end_idx] = 0
        fold_X_train = X[shuffle_idx[train_mask], :]
        _fold_y_train = y[shuffle_idx[train_mask]]
        fold_y_train = np.zeros_like(_fold_y_train)
        tmp_idx = {}
        for idx, _y in enumerate(_fold_y_train):
            if _y not in tmp_idx:
                tmp_idx[_y] = len(tmp_idx)
            fold_y_train[idx] = tmp_idx[_y]

        fold_X_test = X[shuffle_idx[start_idx:end_idx], :]
        _fold_y_test = y[shuffle_idx[start_idx:end_idx]]
        fold_y_test = np.zeros_like(_fold_y_test)
        for idx, _y in enumerate(_fold_y_test):
            if _y not in tmp_idx:
                tmp_idx[_y] = len(tmp_idx)
            fold_y_test[idx] = tmp_idx[_y]

        print("Building model...")
        model = utils.train_logreg_model(fold_X_train, fold_y_train, lambda_l1=0.00001, lambda_l2=0.00001, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        train_accuracies = utils.evaluate_model(model, fold_X_train, fold_y_train, idx2label)
        test_accuracies = utils.evaluate_model(model, fold_X_test, fold_y_test, idx2label)

        overall_train_accuracies.append(train_accuracies['__OVERALL__'])
        overall_test_accuracies.append(test_accuracies['__OVERALL__'])

        # print("Saving everything...")
        # with open(os.path.join(args.output_dir, "model.pkl"), "wb") as fp:
        #     pickle.dump({
        #         'model': model,
        #         'label2idx': label2idx,
        #         'idx2label': idx2label,
        #         'src2idx': src2idx,
        #         'idx2src': idx2src
        #         }, fp)
        
        # with open(os.path.join(args.output_dir, "train_accuracies.json"), "w") as fp:
        #     json.dump(train_accuracies, fp)

        # with open(os.path.join(args.output_dir, "test_accuracies.json"), "w") as fp:
        #     json.dump(test_accuracies, fp)

    print("Overall train accuracy: %0.2f%%"%(100*np.average(overall_train_accuracies)))
    print("Overall test accuracy: %0.2f%%"%(100*np.average(overall_test_accuracies)))

if __name__ == '__main__':
    main()



