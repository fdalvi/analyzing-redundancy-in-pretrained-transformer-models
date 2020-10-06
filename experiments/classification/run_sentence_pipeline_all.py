import argparse
import json
import os
import sys

import numpy as np
import psutil

from collections import defaultdict

# Load aux classifier
if 'AUX_CLASSIFIER_PATH' in os.environ:
    sys.path.append(os.environ['AUX_CLASSIFIER_PATH'])
from aux_classifier import data_loader
from aux_classifier import utils
from aux_classifier import ranking


CONFIGURATION_OPTIONS = {
    # Data directories
    "base_dir": {
        "example": "/some/path/to/files",
        "type": "string",
        "description": "Path to the directory containing data files",
    },
    "train_source": {
        "example": "training_data.word",
        "type": "string",
        "description": "Training tokens file",
    },
    "train_labels": {
        "example": "training_data.labels",
        "type": "string",
        "description": "Training labels file",
    },
    "train_activations": {
        "example": "training_data.hdf5",
        "type": "string",
        "description": "Training activations file",
    },
    "test_source": {
        "example": "test_data.word",
        "type": "string",
        "description": "Test tokens file",
    },
    "test_labels": {
        "example": "test_data.labels",
        "type": "string",
        "description": "Test labels file",
    },
    "test_activations": {
        "example": "test_data.hdf5",
        "type": "string",
        "description": "Test activations file",
    },
    "task_specific_tag": {
        "example": "N",
        "type": "string",
        "description": "Task specific label for unknown labels in test data",
    },
    "output_directory": {
        "example": "path/to/output/directory",
        "type": "string",
        "description": "Path where experiment results should be saved",
    },
    # Experiment variables
    "max_sent_l": {
        "example": 1000,
        "type": "integer",
        "description": "Maximum length of the input sentence",
    },
    "is_brnn": {
        "example": False,
        "type": "boolean",
        "description": "Whether the trained model is a bidirectional model",
    },
    "num_neurons_per_layer": {
        "example": 1024,
        "type": "integer",
        "description": """Number of neurons in every layer of the trained model.
                        Example: BERT: 768, ELMO: 1024""",
    },
    # Ranking variables
    "limit_instances": {
        "example": 40000,
        "type": "integer",
        "description": """Number of instances to limit the training to. Data is sampled
                        proportionally depending on overall class distribution""",
    },
    "clustering_thresholds": {
        "example": [-1, 0.3],
        "type": "list",
        "description": "Thresholds for clustering (Set to -1 for no clustering)",
    },
    "ranking_type": {
        "example": "multiclass",
        "type": "string",
        "description": "Type of ranking to try. Acceptable values: multiclass, binary, multiclasscv, binarycv",
    },
    # Optimization variables
    "num_epochs": {"example": 10, "type": "integer", "description": "Number of epochs"},
    "batch_size": {"example": 128, "type": "integer", "description": "Batch Size"},
    "lambda_l1": {
        "example": 0.00001,
        "type": "float",
        "description": "Regularization L1 parameter",
    },
    "lambda_l2": {
        "example": 0.00001,
        "type": "float",
        "description": "Regularization L2 parameter",
    },
    "model_type": {
        "example": "classification",
        "type": "string",
        "description": "classification/regression",
    },
    "metric": {
        "example": "accuracy",
        "type": "string",
        "description": "accuracy/f1/accuracy_and_f1/pearson/spearman/pearson_and_spearman/matthews_corrcoef",
    },
    # Selection variables
    "performance_deltas": {
        "example": [(3, 1), (2, 1), (1, 1)],
        "type": "list of tuples",
        "description": "Percentage of relative reduction in accuracy allowed while selecting number of layers and choosing minimal neuron set",
    },
}


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def print_sample_configuration():
    props = list(CONFIGURATION_OPTIONS.keys())
    print("{")
    for p in props:
        print(
            "\t%s:%s, # %s"
            % (
                p,
                str(CONFIGURATION_OPTIONS[p]["example"]),
                CONFIGURATION_OPTIONS[p]["description"],
            )
        )
    print("}")


def load_configuration(config_path):
    with open(config_path, "r") as fp:
        config = json.load(fp)
    return config


def is_config_valid(config):
    provided_props = set(config.keys())
    required_props = set(CONFIGURATION_OPTIONS.keys())

    missing_props = list(required_props - provided_props)
    extra_props = list(provided_props - required_props)

    if len(missing_props) > 0:
        print("Some props are missing:")
        print(missing_props)
        return False

    if len(extra_props) > 0:
        print("You included some extra props:")
        print(extra_props)
        return False

    return True


def train_and_eval_model(
    X, y, X_dev, y_dev, X_test, y_test, mappings, config, exp_name, return_model=False
):
    print("Building model...")
    print("Train:", X.shape, y.shape)
    print("Dev:", X_dev.shape, y_dev.shape)
    print("Test:", X_test.shape, y_test.shape)
    if config["model_type"] == "classification":
        label2idx, idx2label, src2idx, idx2src = mappings
        model = utils.train_logreg_model(
            X,
            y,
            lambda_l1=config["lambda_l1"],
            lambda_l2=config["lambda_l2"],
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
        )
        train_scores = utils.evaluate_model(
            model, X, y, metric=config["metric"], idx_to_class=idx2label
        )
        dev_scores = utils.evaluate_model(
            model, X_dev, y_dev, metric=config["metric"], idx_to_class=idx2label
        )
        test_scores = utils.evaluate_model(
            model, X_test, y_test, metric=config["metric"], idx_to_class=idx2label
        )
    elif config["model_type"] == "regression":
        src2idx, idx2src = mappings
        model = utils.train_linear_regression_model(
            X,
            y,
            lambda_l1=config["lambda_l1"],
            lambda_l2=config["lambda_l2"],
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
        )
        train_scores = utils.evaluate_model(model, X, y, metric=config["metric"])
        dev_scores = utils.evaluate_model(model, X_dev, y_dev, metric=config["metric"])
        test_scores = utils.evaluate_model(
            model, X_test, y_test, metric=config["metric"]
        )

    print("Results:")
    print("Train Score (%s): %0.6f" % (config["metric"], train_scores["__OVERALL__"]))
    print("Dev Score (%s): %0.6f" % (config["metric"], dev_scores["__OVERALL__"]))
    print("Test Score (%s): %0.6f" % (config["metric"], test_scores["__OVERALL__"]))

    train_scores = {k: float(v) for k, v in train_scores.items()}
    dev_scores = {k: float(v) for k, v in dev_scores.items()}
    test_scores = {k: float(v) for k, v in test_scores.items()}

    if not return_model:
        return {
            "name": exp_name,
            "train_scores": train_scores,
            "dev_scores": dev_scores,
            "test_scores": test_scores,
        }
    else:
        return (
            {
                "name": exp_name,
                "train_scores": train_scores,
                "dev_scores": dev_scores,
                "test_scores": test_scores,
            },
            model,
        )


def run_baseline(X, y, X_dev, y_dev, X_test, y_test, mappings, config):
    return train_and_eval_model(
        X, y, X_dev, y_dev, X_test, y_test, mappings, config, "baseline"
    )


def run_independent_layerwise(
    X, y, X_dev, y_dev, X_test, y_test, mappings, config, num_layers
):
    all_results = []
    for layer_idx in range(num_layers):
        print(
            "----------------------- Running Layer %02d ------------------------"
            % (layer_idx)
        )
        start_neuron_idx = layer_idx * config["num_neurons_per_layer"]
        end_neuron_idx = (layer_idx + 1) * config["num_neurons_per_layer"]
        selected_neurons = list(range(start_neuron_idx, end_neuron_idx))
        X_filtered = utils.filter_activations_keep_neurons(selected_neurons, X)
        X_dev_filtered = utils.filter_activations_keep_neurons(selected_neurons, X_dev)
        X_test_filtered = utils.filter_activations_keep_neurons(
            selected_neurons, X_test
        )

        all_results.append(
            train_and_eval_model(
                X_filtered,
                y,
                X_dev_filtered,
                y_dev,
                X_test_filtered,
                y_test,
                mappings,
                config,
                "layer-%d" % (layer_idx),
            )
        )
        print(
            "[MEMORY]     Independent layerwise (%02d): %0.2f"
            % (layer_idx, memory_usage_psutil())
        )

    return all_results


def run_incremental_layerwise(
    X, y, X_dev, y_dev, X_test, y_test, mappings, config, num_layers
):
    all_results = []
    for layer_idx in range(num_layers):
        print(
            "-------------------- Running Layer 00 to %02d ---------------------"
            % (layer_idx)
        )
        start_neuron_idx = 0
        end_neuron_idx = (layer_idx + 1) * config["num_neurons_per_layer"]
        selected_neurons = list(range(start_neuron_idx, end_neuron_idx))
        X_filtered = utils.filter_activations_keep_neurons(selected_neurons, X)
        X_dev_filtered = utils.filter_activations_keep_neurons(selected_neurons, X_dev)
        X_test_filtered = utils.filter_activations_keep_neurons(
            selected_neurons, X_test
        )

        all_results.append(
            train_and_eval_model(
                X_filtered,
                y,
                X_dev_filtered,
                y_dev,
                X_test_filtered,
                y_test,
                mappings,
                config,
                "layer-0->%d" % (layer_idx),
            )
        )
        print(
            "[MEMORY]     Incremental layerwise (00 -> %02d): %0.2f"
            % (layer_idx, memory_usage_psutil())
        )

    return all_results


def run_neuron_selection(
    X,
    y,
    X_dev,
    y_dev,
    X_test,
    y_test,
    mappings,
    config,
    num_layers,
    selected_top_layer,
    neuron_selection_delta,
):
    all_results = {}
    start_neuron_idx = 0
    end_neuron_idx = (selected_top_layer + 1) * config["num_neurons_per_layer"]
    selected_neurons = list(range(start_neuron_idx, end_neuron_idx))
    X_layer_filtered = utils.filter_activations_keep_neurons(selected_neurons, X)
    X_dev_layer_filtered = utils.filter_activations_keep_neurons(
        selected_neurons, X_dev
    )
    X_test_layer_filtered = utils.filter_activations_keep_neurons(
        selected_neurons, X_test
    )

    for clustering_threshold in config["clustering_thresholds"]:
        result_key = "clustering-%0.2f" % (clustering_threshold)
        print(
            "------------- Clustering (threshold %0.2f) ----------"
            % (clustering_threshold)
        )
        if clustering_threshold == -1:
            X_filtered = X_layer_filtered
            X_dev_filtered = X_dev_layer_filtered
            X_test_filtered = X_test_layer_filtered
            result_key = "no-clustering"
            all_results[result_key] = {}
        else:
            independent_neurons, clusters = ranking.extract_independent_neurons(
                X_filtered, clustering_threshold=clustering_threshold
            )
            X_filtered = utils.filter_activations_keep_neurons(
                independent_neurons, X_layer_filtered
            )
            X_dev_filtered = utils.filter_activations_keep_neurons(
                independent_neurons, X_dev_layer_filtered
            )
            X_test_filtered = utils.filter_activations_keep_neurons(
                independent_neurons, X_test_layer_filtered
            )
            all_results[result_key] = {}
            all_results[result_key]["clustering_threshold"] = clustering_threshold
            all_results[result_key]["clusters"] = [int(x) for x in clusters]
            all_results[result_key]["independent_neurons"] = [
                int(x) for x in independent_neurons
            ]
            print(
                "Reduced to %d neurons"
                % (len(all_results[result_key]["independent_neurons"]))
            )

        base_results, base_model = train_and_eval_model(
            X_filtered,
            y,
            X_dev_filtered,
            y_dev,
            X_test_filtered,
            y_test,
            mappings,
            config,
            "neuron-selection-base",
            return_model=True,
        )

        all_results[result_key]["base_results"] = base_results
        target_score = all_results[result_key]["base_results"]["dev_scores"]["__OVERALL__"] * (
            1 - neuron_selection_delta / 100.0
        )
        print(
            "Gold score (Baseline with selected top layer [and potential clustering]): %0.6f"
            % (all_results[result_key]["base_results"]["dev_scores"]["__OVERALL__"])
        )
        print(
            "Target score (%d%% drop): %0.6f" % (neuron_selection_delta, target_score)
        )

        print("Getting neuron ranking...")
        if config["model_type"] == "classification":
            label2idx, idx2label, src2idx, idx2src = mappings
            ordering, cutoffs = utils.get_neuron_ordering(
                base_model, label2idx, search_stride=1000
            )
        elif config["model_type"] == "regression":
            src2idx, idx2src = mappings
            ordering, cutoffs = utils.get_neuron_ordering(
                base_model, {"tmp_class": 0}, search_stride=1000
            )

        all_results[result_key]["ordering"] = [int(x) for x in ordering]

        print("Searching for minimal neuron set...")
        # Selection is always done on full neuron set, not filtered
        percentages = [
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]
        minimal_neuron_set_size = X_filtered.shape[1]
        all_results[result_key]["minimal_neuron_set_size"] = minimal_neuron_set_size
        all_results[result_key]["minimal_neuron_set"] = all_results[result_key]["ordering"]
        all_results[result_key]["selected-%d" % (X_filtered.shape[1])] = base_results
        for percentage in percentages:
            selected_num_neurons = int(
                percentage * num_layers * config["num_neurons_per_layer"]
            )

            if selected_num_neurons > X_filtered.shape[1]:
                continue

            print(
                "------------- Retraining with %d neurons (%0.2f%% of total) -------------"
                % (selected_num_neurons, percentage * 100)
            )
            selected_neurons = ordering[:selected_num_neurons]
            X_selected = utils.filter_activations_keep_neurons(
                selected_neurons, X_filtered
            )
            X_dev_selected = utils.filter_activations_keep_neurons(
                selected_neurons, X_dev_filtered
            )
            X_test_selected = utils.filter_activations_keep_neurons(
                selected_neurons, X_test_filtered
            )

            results = train_and_eval_model(
                X_selected,
                y,
                X_dev_selected,
                y_dev,
                X_test_selected,
                y_test,
                mappings,
                config,
                "selected-%d" % (selected_num_neurons),
            )

            all_results[result_key]["selected-%d" % (selected_num_neurons)] = results

            if results["dev_scores"]["__OVERALL__"] > target_score:
                minimal_neuron_set_size = selected_num_neurons
                all_results[result_key][
                    "minimal_neuron_set_size"
                ] = minimal_neuron_set_size
                all_results[result_key]["minimal_neuron_set"] = [
                    int(x) for x in selected_neurons
                ]
                break

    return all_results


def main():
    print("[MEMORY] Begin: %0.2f" % (memory_usage_psutil()))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", dest="config_path", help="Path to configuration file"
    )

    args = parser.parse_args()

    if not args.config_path:
        print(
            "Please provide an experiment configuration file, a sample is shown below:"
        )
        print_sample_configuration()
        sys.exit()

    config = load_configuration(args.config_path)
    if not is_config_valid(config):
        sys.exit()

    all_results = {}

    # setting print option to infinity to print large np arrays instead of ...
    np.set_printoptions(threshold=np.inf)

    train_source_path = os.path.join(config["base_dir"], config["train_source"])
    train_labels_path = os.path.join(config["base_dir"], config["train_labels"])
    train_activations_path = os.path.join(
        config["base_dir"], config["train_activations"]
    )

    # dev_source_path = os.path.join(config["base_dir"], config["dev_source"])
    # dev_labels_path = os.path.join(config["base_dir"], config["dev_labels"])
    # dev_activations_path = os.path.join(config["base_dir"], config["dev_activations"])

    test_source_path = os.path.join(config["base_dir"], config["test_source"])
    test_labels_path = os.path.join(config["base_dir"], config["test_labels"])
    test_activations_path = os.path.join(config["base_dir"], config["test_activations"])

    print("*********************** LOADING ACTIVATIONS ***********************")
    print("[MEMORY] Before activation loading: %0.2f" % (memory_usage_psutil()))
    train_activations, num_layers = data_loader.load_activations(
        train_activations_path,
        config["num_neurons_per_layer"],
        is_brnn=config["is_brnn"],
    )
    # dev_activations, _ = data_loader.load_activations(
    #     dev_activations_path, config["num_neurons_per_layer"], is_brnn=config["is_brnn"]
    # )
    test_activations, _ = data_loader.load_activations(
        test_activations_path,
        config["num_neurons_per_layer"],
        is_brnn=config["is_brnn"],
    )
    print("Number of train sentences: %d" % (len(train_activations)))
    print("Number of test sentences: %d" % (len(test_activations)))
    print("[MEMORY] After activation loading: %0.2f" % (memory_usage_psutil()))

    print("************************* LOADING TOKENS **************************")
    train_tokens = data_loader.load_sentence_data(
        train_source_path, train_labels_path, train_activations
    )
    # dev_tokens = data_loader.load_sentence_data(
    #     dev_source_path, dev_labels_path, dev_activations
    # )
    test_tokens = data_loader.load_sentence_data(
        test_source_path, test_labels_path, test_activations
    )
    print("[MEMORY] After token loading: %0.2f" % (memory_usage_psutil()))

    print("************************ CREATING TENSORS *************************")
    print("Train:")
    X_full, y_full, mappings = utils.create_tensors(
        train_tokens,
        train_activations,
        config["task_specific_tag"],
        model_type=config["model_type"],
    )
    
    # X_dev, y_dev, mappings = utils.create_tensors(
    #     dev_tokens,
    #     dev_activations,
    #     config["task_specific_tag"],
    #     mappings,
    #     model_type=config["model_type"],
    # )
    np.random.seed(7361)
    all_idx = np.arange(X_full.shape[0])
    np.random.shuffle(all_idx)
    num_train_examples = int(X_full.shape[0] * 0.90) # Select 90% for training
    train_idx = all_idx[:num_train_examples]
    dev_idx = all_idx[num_train_examples:]

    all_results['data_split'] = {
        'all_idx': [int(zz) for zz in all_idx],
        'train_idx': [int(zz) for zz in train_idx],
        'dev_idx': [int(zz) for zz in dev_idx]
    }
    
    X = X_full[train_idx, :]
    y = y_full[train_idx]

    X_dev = X_full[dev_idx, :]
    y_dev = y_full[dev_idx]

    print(X.shape, y.shape)
    print("Dev:")
    print(X_dev.shape, y_dev.shape)
    print("Test:")
    X_test, y_test, mappings = utils.create_tensors(
        test_tokens,
        test_activations,
        config["task_specific_tag"],
        mappings,
        model_type=config["model_type"],
    )
    print(X_test.shape, y_test.shape)
    print("[MEMORY] After tensor creation: %0.2f" % (memory_usage_psutil()))

    print("************************* FREEING MEMORY **************************")
    import gc

    train_tokens["source"] = None
    train_tokens["target"] = None
    train_tokens = None

    for idx, _ in enumerate(train_activations):
        train_activations[idx] = None
    train_activations = None

    # dev_tokens["source"] = None
    # dev_tokens["target"] = None
    # dev_tokens = None

    # for idx, _ in enumerate(dev_activations):
    #     dev_activations[idx] = None
    # dev_activations = None

    test_tokens["source"] = None
    test_tokens["target"] = None
    test_tokens = None
    for idx, _ in enumerate(test_activations):
        test_activations[idx] = None
    test_activations = None

    gc.collect()
    print("[MEMORY] After cleanup: %0.2f" % (memory_usage_psutil()))

    print("***************** RUNNING BASELINE [ ALL LAYERS ] *****************")
    results = run_baseline(X, y, X_dev, y_dev, X_test, y_test, mappings, config)
    all_results["baseline"] = results
    print("[MEMORY] After baseline run: %0.2f" % (memory_usage_psutil()))

    print("****************** RUNNING INDEPENDENT LAYERWISE ******************")
    results = run_independent_layerwise(
        X, y, X_dev, y_dev, X_test, y_test, mappings, config, num_layers
    )
    all_results["independent_layerwise"] = results
    print("[MEMORY] After indendent layerwise run: %0.2f" % (memory_usage_psutil()))

    print("****************** RUNNING INCREMENTAL LAYERWISE ******************")
    # Only run until penultimate layer since baseline is same as last incremental
    # layerwise result
    results = run_incremental_layerwise(
        X, y, X_dev, y_dev, X_test, y_test, mappings, config, num_layers - 1
    )
    layer_idx = num_layers - 1
    print(
        "---------------- Copying Layer 00 to %02d from baseline -----------------"
        % (layer_idx)
    )
    results.append(
        {
            "name": "layer-%d" % (layer_idx),
            "train_scores": all_results["baseline"]["train_scores"],
            "dev_scores": all_results["baseline"]["dev_scores"],
            "test_scores": all_results["baseline"]["test_scores"],
        }
    )
    all_results["incremental_layerwise"] = results
    print("[MEMORY] After incremental layerwise run: %0.2f" % (memory_usage_psutil()))

    print("********************* SELECTING NEURONS *********************")
    all_results["selection"] = {}
    for selection_paramaters in config["performance_deltas"]:
        layer_selection_delta, neuron_selection_delta = selection_paramaters
        result_key = "selection-%d-%d" % (layer_selection_delta, neuron_selection_delta)
        print(
            "********************* SELECTING MINIMUM LAYERS (%d%% drop) *********************"
            % (layer_selection_delta)
        )
        selected_top_layer = num_layers - 1
        target_score = all_results["baseline"]["dev_scores"]["__OVERALL__"] * (
            1 - layer_selection_delta / 100.0
        )
        print(
            "Gold score (%s) (Baseline): %0.6f"
            % (config["metric"], all_results["baseline"]["dev_scores"]["__OVERALL__"])
        )
        print(
            "Target score (%s) (%d%% drop): %0.6f"
            % (config["metric"], layer_selection_delta, target_score)
        )
        for layer_idx in range(num_layers):
            if (
                all_results["incremental_layerwise"][layer_idx]["dev_scores"][
                    "__OVERALL__"
                ]
                > target_score
            ):
                selected_top_layer = layer_idx
                break
        print("Selected top layer: Layer %02d" % (selected_top_layer))
        print(
            "Dev score: %0.6f"
            % (
                all_results["incremental_layerwise"][selected_top_layer]["dev_scores"][
                    "__OVERALL__"
                ]
            )
        )

        all_results["selection"][result_key] = {}
        all_results["selection"][result_key]["selected_top_layer"] = {
            "selected_top_layer": selected_top_layer,
            "train_score": all_results["incremental_layerwise"][selected_top_layer][
                "train_scores"
            ]["__OVERALL__"],
            "dev_score": all_results["incremental_layerwise"][selected_top_layer][
                "dev_scores"
            ]["__OVERALL__"],
            "test_score": all_results["incremental_layerwise"][selected_top_layer][
                "test_scores"
            ]["__OVERALL__"],
        }
        print("[MEMORY] After top layer selection: %0.2f" % (memory_usage_psutil()))

        print("********************* RUNNING NEURON SELECTION *********************")
        results = run_neuron_selection(
            X,
            y,
            X_dev,
            y_dev,
            X_test,
            y_test,
            mappings,
            config,
            num_layers,
            selected_top_layer,
            neuron_selection_delta,
        )
        all_results["selection"][result_key]["neuron_selection"] = results
        print("[MEMORY] After neuron selection: %0.2f" % (memory_usage_psutil()))

    print("***************************** SUMMARY *****************************")
    print(
        "Baseline Score (%s): %0.6f"
        % (config["metric"], all_results["baseline"]["test_scores"]["__OVERALL__"])
    )
    print("Independent Layerwise:")
    for layer_idx in range(num_layers):
        print(
            "  Layer %02d: %0.6f"
            % (
                layer_idx,
                all_results["independent_layerwise"][layer_idx]["test_scores"][
                    "__OVERALL__"
                ],
            )
        )
    print("Incremental Layerwise:")
    for layer_idx in range(num_layers):
        print(
            "  Layer 0 -> %02d: %0.6f"
            % (
                layer_idx,
                all_results["incremental_layerwise"][layer_idx]["test_scores"][
                    "__OVERALL__"
                ],
            )
        )
    for layer_result_key in all_results["selection"]:
        print("Selection parameters:", layer_result_key)
        print(
            "Selected top layer: Layer %02d"
            % (all_results["selection"][layer_result_key]["selected_top_layer"]["selected_top_layer"])
        )

        for neuron_result_key in all_results["selection"][layer_result_key][
            "neuron_selection"
        ]:
            print("Clustering paramter:", neuron_result_key)
            neuron_selection_results = all_results["selection"][layer_result_key][
                "neuron_selection"
            ][neuron_result_key]
            if neuron_result_key != "no-clustering":
                print(
                    "Independent Neurons: %d neurons"
                    % (len(neuron_selection_results["independent_neurons"]))
                )
                print(
                    "CC score (%s): %0.6f"
                    % (
                        config["metric"],
                        neuron_selection_results["base_results"]["test_scores"][
                            "__OVERALL__"
                        ],
                    )
                )
            selected_num_neurons = neuron_selection_results["minimal_neuron_set_size"]
            print("Minimal set of neurons: %d neurons" % (selected_num_neurons))
            print(
                "Minimal set score (%s): %0.6f"
                % (
                    config["metric"],
                    neuron_selection_results["selected-%d" % (selected_num_neurons)][
                        "test_scores"
                    ]["__OVERALL__"],
                )
            )

    with open(os.path.join(config["output_directory"], "all_results.json"), "w") as fp:
        fp.write(json.dumps(all_results))


if __name__ == "__main__":
    main()
