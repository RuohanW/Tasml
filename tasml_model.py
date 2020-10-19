# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Code defining LEO inner loop.

See "Meta-Learning with Latent Embedding Optimization" by Rusu et al.
(https://arxiv.org/pdf/1807.05960.pdf).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

import data_new as data_module

import data as data_unlimited

NDIM = 640

class LeastSquareMeta(snt.AbstractModule):
    """Sonnet module implementing the inner loop of LEO."""

    def __init__(self, layers, lam, num_classes, name="maml", limited=True, l2_weight=1e-8, t_weight=1.):
        super(LeastSquareMeta, self).__init__(name=name)
        self.latent_size = layers[-1]
        self.lam = lam
        self.num_classes = num_classes
        self.layers = layers
        self._l2_penalty_weight = l2_weight
        self.limited = limited
        self.t_weight = t_weight


    def _build(self, data, is_meta_training=True, test_data=None):
        if isinstance(data, list):
            if self.limited:
                data = data_module.ProblemInstance(*data)
            else:
                data = data_unlimited.ProblemInstance(*data)

        if test_data is not None:
            test_latent = self.encoder(test_data.tr_input)

            tr_loss, tr_acc = self.least_square(test_data, test_latent, use_val=False)

        self.is_meta_training = is_meta_training
        self.save_problem_instance_stats(data.tr_input)

        latents = self.encoder(data.tr_input)
        val_loss, val_accuracy = self.least_square(data, latents)


        batch_val_loss = tf.reduce_mean(val_loss)
        if self.limited:
            batch_val_loss *= data.weight
        batch_val_accuracy = tf.reduce_mean(val_accuracy)
        regularization_penalty = self._l2_regularization


        if test_data is not None:
            batch_val_loss += tr_loss * self.t_weight

        return batch_val_loss, regularization_penalty, batch_val_accuracy

    @snt.reuse_variables
    def least_square(self, data, latent, use_val=True):
        with tf.variable_scope("maml_inner"):
            X = tf.reshape(latent, [-1, self.latent_size])
            X_t = tf.transpose(X)

            A = self.lam * tf.eye(tf.shape(X)[0]) + tf.matmul(X, X_t)
            tr_out = tf.reshape(data.tr_output, [-1])
            out_one_hot = tf.one_hot(tr_out, self.num_classes)
            tmp = tf.matmul(X_t, tf.linalg.inv(A))
            weights = tf.matmul(tmp, out_one_hot)

            if use_val:
                X_test = tf.reshape(data.val_input, [-1, NDIM])

                test_latent = self.encoder(X_test)

                pred = tf.matmul(test_latent, weights)

                test_out = tf.reshape(data.val_output, [-1])
            else:
                pred = tf.matmul(X, weights)

                test_out = tr_out

            out_one_hot = tf.one_hot(test_out, self.num_classes)
            loss = tf.reduce_mean(tf.square(pred - out_one_hot))
            c = tf.argmax(pred, axis=1, output_type=tf.int32)

            acc = tf.contrib.metrics.accuracy(test_out, c)
            return loss, acc

    @snt.reuse_variables
    def encoder(self, inputs):
        X = tf.reshape(inputs, [-1, inputs.shape[-1]])
        with tf.variable_scope("encoder"):
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=tf.float32)
            encoder_module = snt.nets.MLP(
                self.layers,
                use_bias=False,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            outputs = encoder_module(X)
            outputs = X + outputs
            return outputs

    def grads_and_vars(self, metatrain_loss):
        metatrain_variables = self.trainable_variables
        metatrain_gradients = tf.gradients(metatrain_loss, metatrain_variables)

        nan_loss_or_grad = tf.logical_or(
            tf.is_nan(metatrain_loss),
            tf.reduce_any([tf.reduce_any(tf.is_nan(g))
                           for g in metatrain_gradients]))


        regularization_penalty = (1e-4 * self._l2_regularization)
        zero_or_regularization_gradients = [
            g if g is not None else tf.zeros_like(v)
            for v, g in zip(tf.gradients(regularization_penalty,
                                         metatrain_variables), metatrain_variables)]

        metatrain_gradients = tf.cond(nan_loss_or_grad,
                                      lambda: zero_or_regularization_gradients,
                                      lambda: metatrain_gradients, strict=True)

        return metatrain_gradients, metatrain_variables

    @property
    def _l2_regularization(self):
        return tf.cast(
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            dtype=tf.float32)

    def save_problem_instance_stats(self, instance):
        num_classes, num_examples_per_class, embedding_dim = instance.get_shape()
        if hasattr(self, "num_classes"):
            assert self.num_classes == num_classes, (
                "Given different number of classes (N in N-way) in consecutive runs.")
        if hasattr(self, "num_examples_per_class"):
            assert self.num_examples_per_class == num_examples_per_class, (
                "Given different number of examples (K in K-shot) in consecutive"
                "runs.")
        if hasattr(self, "embedding_dim"):
            assert self.embedding_dim == embedding_dim, (
                "Given different embedding dimension in consecutive runs.")

        self.num_classes = num_classes
        self.num_examples_per_class = num_examples_per_class
        self.embedding_dim = embedding_dim