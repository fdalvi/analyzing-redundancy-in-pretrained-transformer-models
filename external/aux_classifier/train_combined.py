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
from tqdm import tqdm, tqdm_notebook, tnrange

# Import lib
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import aux_classifier.utils as utils
import aux_classifier.representations as repr
import aux_classifier.data_loader as data_loader

def main():
    parser = argparse.ArgumentParser(description='Train a classifier')
    
    parser.add_argument('--train-labels', dest='train_labels', required=True,
                    help='Location of train source labels')    
    parser.add_argument('--test-labels', dest='test_labels', required=True,
                    help='Location of test source labels')
    

    parser.add_argument('--word-train-source', dest='train_source', required=True,
                    help='Location of train source file')
    parser.add_argument('--word-train-activations', dest='train_activations', required=True,
                    help='Location of train source activations')
    parser.add_argument('--word-test-source', dest='test_source', required=True,
                    help='Location of test source file')
    parser.add_argument('--word-test-activations', dest='test_activations', required=True,
                    help='Location of test source activations')

    parser.add_argument('--bpe-train-source', dest='bpe_train_source',
                    help='Location of aux train source file (BPE/CHAR)')
    parser.add_argument('--bpe-train-activations', dest='bpe_train_activations', required=True,
                    help='Location of test source activations')
    parser.add_argument('--bpe-test-source', dest='bpe_test_source',
                    help='Location of aux test source file (BPE/CHAR)')
    parser.add_argument('--bpe-test-activations', dest='bpe_test_activations', required=True,
                    help='Location of test source activations')

    parser.add_argument('--char-train-source', dest='char_train_source',
                    help='Location of aux train source file (BPE/CHAR)')
    parser.add_argument('--char-train-activations', dest='char_train_activations', required=True,
                    help='Location of test source activations')
    parser.add_argument('--char-test-source', dest='char_test_source',
                    help='Location of aux test source file (BPE/CHAR)')
    parser.add_argument('--char-test-activations', dest='char_test_activations', required=True,
                    help='Location of test source activations')


    parser.add_argument('--task-specific-tag', dest='task_specific_tag', 
                    required=True, help='Tag incase test has unknown tags')

    parser.add_argument('--max-sent-l', dest='max_sent_l', type=int,
                    default=1000, help='Maximum sentence length')
    parser.add_argument('--is-bidirectional', dest='is_brnn', type=bool,
                    default=True, help='Set to false if original model is unidirectional, \
                                or if the representations are from the decoder side')

    parser.add_argument('--output-dir', dest='output_dir', 
                    required=True, help='Location to save all results')

    parser.add_argument('--filter-layers', dest='filter_layers', default=None,
                    type=str, help='Use specific layers for training. Format: f1,b1,f2,b2')

    args = parser.parse_args()

    # Constants
    NUM_EPOCHS = 10
    BATCH_SIZE = 512

    print("Loading activations...")
    train_activations, NUM_LAYERS = data_loader.load_activations(args.train_activations, 500)
    test_activations, _ = data_loader.load_activations(args.test_activations, 500)

    bpe_train_activations, _ = data_loader.load_activations(args.bpe_train_activations, 500)
    bpe_test_activations, _ = data_loader.load_activations(args.bpe_test_activations, 500)

    char_train_activations, _ = data_loader.load_activations(args.char_train_activations, 500)
    char_test_activations, _ = data_loader.load_activations(args.char_test_activations, 500)

    print("Number of train sentences: %d"%(len(train_activations)))
    print("Number of test sentences: %d"%(len(test_activations)))

    print("Loading word data...")
    train_tokens = data_loader.load_data(args.train_source, args.train_labels, train_activations, 250)
    test_tokens = data_loader.load_data(args.test_source, args.test_labels, test_activations, 250)
    
    print("Loading BPE data...")
    bpe_train_tokens = data_loader.load_aux_data(args.train_source, args.train_labels, args.bpe_train_source, bpe_train_activations, 250)
    bpe_test_tokens = data_loader.load_aux_data(args.test_source, args.test_labels, args.bpe_test_source, bpe_test_activations, 250)

    print("Loading Char data...")
    char_train_tokens = data_loader.load_aux_data(args.train_source, args.train_labels, args.char_train_source, char_train_activations, args.max_sent_l)
    char_test_tokens = data_loader.load_aux_data(args.test_source, args.test_labels, args.char_test_source, char_test_activations, args.max_sent_l)

    NUM_TOKENS = sum([len(t) for t in train_tokens['target']])
    print('Number of total train tokens: %d'%(NUM_TOKENS))

    NUM_SOURCE_TOKENS = sum([len(t) for t in train_tokens['source']])
    print('Number of source words: %d'%(NUM_SOURCE_TOKENS)) 

    NUM_NEURONS = train_activations[0].shape[1]
    print('Number of neurons: %d'%(NUM_NEURONS))

    print("Processing BPE activations...")
    bpe_train_activations = repr.bpe_get_last_activations(bpe_train_tokens, bpe_train_activations, is_brnn=args.is_brnn)
    bpe_test_activations = repr.bpe_get_last_activations(bpe_test_tokens, bpe_test_activations, is_brnn=args.is_brnn)

    print("Processing Char activations...")
    char_train_activations = repr.char_get_last_activations(char_train_tokens, char_train_activations, is_brnn=args.is_brnn)
    char_test_activations = repr.char_get_last_activations(char_test_tokens, char_test_activations, is_brnn=args.is_brnn)
    
    print("Creating train tensors...")
    print("Word...")
    word_X, y, mappings = utils.create_tensors(train_tokens, train_activations, args.task_specific_tag)
    print("BPE...")
    bpe_X, _, mappings = utils.create_tensors(train_tokens, bpe_train_activations, args.task_specific_tag)
    print("Char...")
    char_X, _, mappings = utils.create_tensors(train_tokens, char_train_activations, args.task_specific_tag)

    print("Creating test tensors...")
    print("Word...")
    word_X_test, y_test, mappings = utils.create_tensors(test_tokens, test_activations, args.task_specific_tag, mappings)
    print("BPE...")
    bpe_X_test, _, _ = utils.create_tensors(test_tokens, bpe_test_activations, args.task_specific_tag, mappings)
    print("Char...")
    char_X_test, _, _ = utils.create_tensors(test_tokens, char_test_activations, args.task_specific_tag, mappings)

    label2idx, idx2label, src2idx, idx2src = mappings

    experiments = [
        ('word',[word_X], [word_X_test]),
        ('bpe',[bpe_X], [bpe_X_test]),
        ('char',[char_X], [char_X_test]),
        ('word+bpe',[word_X, bpe_X], [word_X_test, bpe_X_test]),
        ('bpe+char',[bpe_X, char_X], [bpe_X_test, char_X_test]),
        ('word+char',[word_X, char_X], [word_X_test, char_X_test]),
        ('word+bpe+char',[word_X, bpe_X, char_X], [word_X_test, bpe_X_test, char_X_test]),
    ]

    accs = []

    for exp in experiments:
        exp_name, trains, tests = exp
        print("Running %s..."%(exp_name))
        print("Building model...")
        X = np.concatenate(trains, axis=1)
        X_test = np.concatenate(tests, axis=1)
        print(X.shape, [x.shape for x in trains])
        model = utils.train_logreg_model(X, y, lambda_l1=0.00001, lambda_l2=0.00001, num_epochs=10, batch_size=512)
        train_accuracies = utils.evaluate_model(model, X, y, idx2label)
        test_accuracies, predictions = utils.evaluate_model(model, X_test, y_test, idx2label, return_predictions=True, source_tokens=test_tokens['source'])

        print("Saving everything...")
        out = os.path.join(args.output_dir, exp_name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "model.pkl"), "wb") as fp:
            pickle.dump({
                'model': model,
                'label2idx': label2idx,
                'idx2label': idx2label,
                'src2idx': src2idx,
                'idx2src': idx2src
                }, fp)
    
        with open(os.path.join(out, "train_accuracies.json"), "w") as fp:
            json.dump(train_accuracies, fp)

        with open(os.path.join(out, "test_accuracies.json"), "w") as fp:
            json.dump(test_accuracies, fp)

        with open(os.path.join(out, "test_predictions.json"), "w") as fp:
            json.dump(predictions, fp)

        accs += [(exp_name, train_accuracies['__OVERALL__'], test_accuracies['__OVERALL__'])]

    print(accs)

if __name__ == '__main__':
    main()



