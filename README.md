# Analyzing Redundancy in Pretrained Transformer Models

Code and instructions for the paper titled **Analyzing Redundancy in Pretrained Transformer Models** published at [EMNLP 2020](https://2020.emnlp.org). This codebase
extensively uses several other toolkits, including:

* [contextual-repr-analysis](https://github.com/nelson-liu/contextual-repr-analysis) for embedding extraction for sequence labelling tasks
* [transformers](https://github.com/huggingface/transformers) for embedding extraction for sequence classification tasks
* [aux_classifier](https://github.com/fdalvi/aux_classifier) for all layer and neuron level analysis

## Getting the Data
### Sequence Labeling
The datasets for sequence labeling tasks can be downloaded from the following sources:

- **POS** - https://catalog.ldc.upenn.edu/LDC99T42
- **SEM** - https://pmb.let.rug.nl/data.php
- **CCG Supertagging** - https://catalog.ldc.upenn.edu/LDC2005T13
- **Chunking** - https://www.clips.uantwerpen.be/conll2000/chunking/

We use the first 150,000 tokens from the train set for all of these tasks, and standard development and test sets.

### Sequence Classification
We use the [GLUE Benchmark](https://gluebenchmark.com/) for all sequence classification experiments. Download the data from https://gluebenchmark.com/tasks. For the experiments in this paper, you should download and have the following size directories, each containing the train and dev `*.tsv`'s: SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE. Run `data/process_sentence_data.sh` to the base data directory to process and create additional required files.

We split the *official train* sets into train and development internally (provided code automatically does this at runtime with a fixed seed for reproducibility), using 90% as train and 10% as development. We use the *official development* sets as our test internally, since the *official test* sets are not publicly available.

## Extracting Activations
### Sequence Labeling
We use the [contextual-repr-analysis](https://github.com/nelson-liu/contextual-repr-analysis) by Lui. et al. for handling the sequence labeling data for POS, SEM, CCG and Chunking and extracting their activations.

### Sequence Classification
We use the [transformers](https://github.com/huggingface/transformers) library to extract activations for sequences for all of our tasks. Specifically, we extract the activations of the [CLS] token from each layer. The exact `transformers` code used is included in the repo in `external/transformers`.

1. Create `conda` environment to install required dependencies:

```bash
cd external/transformers
conda create --name transformers python=3
conda activate transformers
pip install -r requirements.txt
```

2. Use `external/transformers/examples/run_glue_extraction.py` to extract the activations:

```bash
TRANSFORMERS_PATH=<full-path-to-external/transformers> \
    python run_glue_extraction.py \
        --data_dir <data-dir> \
        --model_type <model-arch> \
        --model_name_or_path <path-to-finetuned-model> \
        --task_name <task-name> \
        --output_file <output-json> \
        --cache_dir <cache-dir> \
        --do_train \
        --do_lower_case \
        --per_gpu_batch_size 32 \
        --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
        --sentence_only
```
where
- `<full-path-to-external/transformers>` is the path to the included transformers in this repo
- `<data-dir>` is the data for a specific task downloaded earlier (`data/sst-2` for example)
- `<model-arch>` is `bert` or `xlnet`
- `<path-to-finetuned-model>` is the path to a model finetuned on the specific task. Finetuning is done is the standard way as described in the `transformers` package, using `run_glue.py`
- `<task-name>` is the name of a task (e.g. `sst-2`)
- `<output-file>` is the path to output embeddings in json
- [optional] `<cache-dir>` is a custom transformers cache

## Running Experiments
All of the experiments are primarily run using the provided `external/aux_classifier` code. To setup, you will need to create a `conda` environment for the dependencies:

```bash
cd external/aux_classifier
conda create -f aux_classifier_env.yml
conda activate aux_classifier
```

### Sequence Labeling
To run sequence labeling experiments, use the following command:

```bash
AUX_CLASSIFIER_PATH=<full-path-to-external/aux_classifier> \
    python run_pipeline_all.py --config <exp-config-json>
```
where
- `<full-path-to-external/aux_classifier>` is the path to the included aux_classifier in this repo
- `<exp-config-json>` is the configuration for the current experiment, an example configuration is provided in `experiments/labeling/example_config.json`, and a detailed description is printed if `run_pipeline_all.py` is run without any arguments

Three helper scripts are provided:
1. `experiments/labeling/run_pipeline_all.py`: This produces oracle numbers, individual classifier layer numbers, and concatenated classifier numbers. Given a list of *correlation clustering coefficients* and *performance deltas* for `LayerSelector` and `CCFS`, also produces the corresponding accuracies (Used in Section 6.1 and 7 in the paper).
2. `experiments/labeling/run_cc_all.py`: This produces oracle numbers and performance numbers at all *correlation clustering* thresholds (Used in Section 5.2).
3. `experiments/labeling/run_max_features.py`: This produces oracle numbers and minimal set of neurons from all neurons accuracies (Used in Section 6.2).

### Sequence Classification
To run sequence classification experiments, use the following command:

```bash
AUX_CLASSIFIER_PATH=<full-path-to-external/aux_classifier> \
    python run_sentence_pipeline_all.py --config <exp-config-json>
```
where
- `<full-path-to-external/aux_classifier>` is the path to the included aux_classifier in this repo
- `<exp-config-json>` is the configuration for the current experiment, an example configuration is provided in `experiments/classification/example_config.json`, and a detailed description is printed if `run_sentence_pipeline_all.py` is run without any arguments

Three helper scripts are provided:
1. `experiments/classification/run_sentence_pipeline_all.py`: This produces oracle numbers, individual classifier layer numbers, and concatenated classifier numbers. Given a list of *correlation clustering coefficients* and *performance deltas* for `LayerSelector` and `CCFS`, also produces the corresponding accuracies (Used in Section 6.1 and 7 in the paper).
2. `experiments/classification/run_sentence_cc_all.py`: This produces oracle numbers and performance numbers at all *correlation clustering* thresholds (Used in Section 5.2).
3. `experiments/classification/run_sentence_max_features.py`: This produces oracle numbers and minimal set of neurons from all neurons accuracies (Used in Section 6.2).

## Running Analysis Code
- The clustering analysis presented in Section 5.2 can be reproduced by running `experiments/cluster_analysis.py`

- The timing analysis presented in Section 7 can be reproduced using the following scripts:
    1. Pretrained model extraction timing: `external/transformers/examples/run_glue_timing.py`, using similar arguments as the extraction scripts
    2. Classifier training timing: `experiments/classifier_timing_analysis.py`

## Computing Infrastructure
All experiments were run on machines with 6-core 2.8 GHz AMD Opteron Processor 4184 processors, NVidia GeForce GTX TITAN X graphics cards and 128GB of RAM.

## Evaluation Metrics
We used `accuracy` measure for all experiments, except for STS-B, where `matthews_corrcoef` was used. The metrics were computed using the [scikit-learn](https://scikit-learn.org) library, with detailed code available in `external/aux_classifier/aux_classifier/metrics.py`

## Number of Parameters in used models
Pretrained models:
- BERT: Standard `bert-base` with 110M parameters
- XLNet: Standard `xlnet-base` with 116M parameters

Trained classifiers:
- Linear classifiers with number of parameters roughly equal to `num_of_input_features` x `num_of_classes`. In the worst case, we use all 9984 features extracted from the pre-trained models, and our proposed algorithm reduces this number by a significant amount.

## Approximate runtimes
- Pretrained models:
    - BERT: 0.715 ms per instance
    - XLNet: 1.246 ms per instance
- Classifier:
    - Full 9984 feature set with 100,000K input instances, 10 epochs and 2 output classes: 48.48 seconds

## Hyperparameters
### Pretrained Models
We use the default hyperparameters provided by the transformers library. They are listed below just as additional information:

```
Optimizer parameters (Adam)
===========================
adam_epsilon=1e-08
gradient_accumulation_steps=1
learning_rate=5e-05
max_grad_norm=1.0
max_seq_length=128
num_train_epochs=3.0
per_gpu_eval_batch_size=8
per_gpu_train_batch_size=8
seed=42
warmup_steps=0
weight_decay=0.0

Model parameters (BERT)
=======================
transformers-model: bert-base-cased
attention_probs_dropout_prob: 0.1
hidden_act: gelu
hidden_dropout_prob: 0.1
hidden_size: 768
initializer_range: 0.02
intermediate_size: 3072
layer_norm_eps: 1e-12
max_position_embeddings: 512
num_attention_heads: 12
num_hidden_layers: 12
vocab_size: 30522

Model parameters (XLNet)
========================
transformers-model: xlnet-base-cased
attn_type: "bi"
bi_data: false
clamp_len: -1
d_head: 64
d_inner: 3072
d_model: 768
dropout: 0.1
end_n_top: 5
ff_activation: "gelu"
initializer_range: 0.02
layer_norm_eps: 1e-12
n_head: 12
n_layer: 12
n_token: 32000
start_n_top: 5
summary_activation: "tanh"
summary_last_dropout: 0.1
summary_type: "last"
summary_use_proj: true
```
### Classifier Models
The default hyperparameters have been set in the provided code, but are listed here for information:
```
Classifier: Logistic Regression w/ Elastic Net Regularization
Epochs: 10
Learning rate: 1e-3
Batch size: 128
L1 regularization: 1e-5
L2 regularization: 1e-5
```
