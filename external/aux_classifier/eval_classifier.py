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

def load_data_and_train(model_path, test_source, test_aux_source, test_labels, test_activations,
                        exp_type, task_specific_tag, max_sent_l, batch_size,
                        is_brnn, filter_layers):
    print("Loading activations...")
    test_activations = data_loader.load_activations(test_activations)
    print("Number of test sentences: %d"%(len(test_activations)))

    if exp_type == 'word' or exp_type == 'charcnn':
        test_tokens = data_loader.load_data(test_source, test_labels, test_activations, max_sent_l)
    else:
        test_tokens = data_loader.load_aux_data(test_source, test_labels, test_aux_source, test_activations, max_sent_l)

    if exp_type == 'bpe_avg':
        test_activations = repr.bpe_get_avg_activations(test_tokens, test_activations)
    elif exp_type == 'bpe_last':
        test_activations = repr.bpe_get_last_activations(test_tokens, test_activations, is_brnn=is_brnn)
    elif exp_type == 'char_avg':
        test_activations = repr.char_get_avg_activations(test_tokens, test_activations)
    elif exp_type == 'char_last':
        test_activations = repr.char_get_last_activations(test_tokens, test_activations, is_brnn=is_brnn)

    # Filtering
    if filter_layers:
        assert False, "filtering not yet supported"
        train_activations, test_activations = utils.filter_activations_by_layers(train_activations, test_activations, filter_layers, 500, 2)

    print("Loading trained model...")
    with open(model_path, "rb") as fp:
        model_parts = pickle.load(fp)

    model = model_parts['model']
    label2idx = model_parts['label2idx']
    idx2label = model_parts['idx2label']
    src2idx = model_parts['src2idx']
    idx2src = model_parts['idx2src']
    mappings = label2idx, idx2label, src2idx, idx2src

    print("Creating test tensors...")
    X_test, y_test, mappings = utils.create_tensors(test_tokens, test_activations, task_specific_tag, mappings)

    label2idx, idx2label, src2idx, idx2src = mappings

    print("Building model...")
    test_accuracies, predictions = utils.evaluate_model(model, X_test, y_test, idx2label, return_predictions=True, source_tokens=test_tokens['source'])

    return test_accuracies, predictions, test_tokens

def main():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--model-path', dest='model_path', required=True,
                    help='Location of saved model (*.pkl)')

    parser.add_argument('--test-source', dest='test_source', required=True,
                    help='Location of test source file')
    parser.add_argument('--test-aux-source', dest='test_aux_source',
                    help='Location of aux test source file (BPE/CHAR)')
    parser.add_argument('--test-labels', dest='test_labels', required=True,
                    help='Location of test source labels')
    parser.add_argument('--test-activations', dest='test_activations', required=True,
                    help='Location of test source activations')
    
    parser.add_argument('--exp-type', dest='exp_type', 
                    choices=['word', 'charcnn', 'bpe_avg', 'bpe_last', 'char_avg', 'char_last'],
                    default='word', required=True,
                    help='Type of experiment')

    parser.add_argument('--task-specific-tag', dest='task_specific_tag', 
                    required=True, help='Tag incase test has unknown tags')

    parser.add_argument('--max-sent-l', dest='max_sent_l', type=int,
                    default=250, help='Maximum sentence length')
    parser.add_argument('--is-bidirectional', dest='is_brnn', type=bool,
                    default=True, help='Set to false if original model is unidirectional, \
                                or if the representations are from the decoder side')

    parser.add_argument('--output-dir', dest='output_dir', 
                    required=True, help='Location to save all results')

    parser.add_argument('--filter-layers', dest='filter_layers', default=None,
                    type=str, help='Use specific layers for training. Format: f1,b1,f2,b2')

    args = parser.parse_args()

    print("Creating output directory...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Constants
    BATCH_SIZE = 512

    result = load_data_and_train(args.model_path, args.test_source, args.test_aux_source, args.test_labels, args.test_activations,
                        args.exp_type, args.task_specific_tag, args.max_sent_l, BATCH_SIZE,
                        args.is_brnn, args.filter_layers)

    test_accuracies, test_predictions, test_tokens = result

    print("Saving everything...")
    with open(os.path.join(args.output_dir, "test_accuracies.json"), "w") as fp:
        json.dump(test_accuracies, fp)

    with open(os.path.join(args.output_dir, "test_predictions.json"), "w") as fp:
        json.dump(test_predictions, fp)

if __name__ == '__main__':
    main()



