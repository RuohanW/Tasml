# Meta-Learning with Latent Embedding Optimization

## Overview
This repository contains the implementation of the meta-learning model
described in the paper "[Structured Prediction for Conditional Meta-Learning](https://arxiv.org/abs/2002.08799)". It will be presented at NeurIPS 2020.

The paper learns a task-conditional meta-parameters using "[structured prediction](https://en.wikipedia.org/wiki/Structured_prediction)", which considers the meta-parameters as structured output.

The code uses the same pre-trained [embedding](http://storage.googleapis.com/leo-embeddings/embeddings.zip) as provided in Google's [LEO paper](https://github.com/deepmind/leo). 

## Running the code

### Setup
To run the code, you first need to need to install:

- [TensorFlow](https://www.tensorflow.org/install/) and [TensorFlow Probability](https://www.tensorflow.org/probability) (we used version 1.13),
- [Sonnet](https://github.com/deepmind/sonnet) (we used version v1.29), and
- [Abseil](https://github.com/abseil/abseil-py) (we use only the FLAGS module).

### Getting the data
The code looks for the extracted embedding directory at `~/workspace/data/embeddings` by default. This could be changed in `config.py`.

### Running the code
Then, clone this repository using:

`$ git clone https://github.com/deepmind/leo`

Step 1: to construct a meta-train set of fixed size and compute the task similarity matrix, run

`$ python runner_tasml.py gen_db`

Step 2 (optional): To train an unconditional meta-learning model for warm-starting structured prediction, run

`$ python runner_tasml.py uncon_meta`

To run structured prediction on 100 targets tasks, given a fixed size meta-train set (Step 1) with warmstart (Step 2), run
`$ python runner_tasml.py sp`

This will train the model for solving 5-way 1-shot miniImageNet classification.
