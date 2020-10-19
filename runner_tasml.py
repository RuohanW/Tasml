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
"""A binary building the graph and performing the optimization of LEO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import pickle

from absl import flags
from six.moves import zip
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import config
import data_new as data
import data as data_unlimited

import tasml_model as model
import utils

import numpy as np

import os.path as osp
import os

from scipy.linalg import cho_factor, cho_solve

chkpoint_root = "checkpoint/save"

save_root = "checkpoint/"


FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", chkpoint_root, "Path to restore from and "
                                                  "save to checkpoints.")
flags.DEFINE_integer("checkpoint_steps", 500, "The frequency, in number of "
                              "steps, of saving the checkpoints.")

flags.DEFINE_boolean("evaluation_mode", False, "Whether to run in an "
                                               "evaluation-only mode.")




def _clip_gradients(gradients, gradient_threshold, gradient_norm_threshold):
    """Clips gradients by value and then by norm."""
    if gradient_threshold > 0:
        gradients = [
            tf.clip_by_value(g, -gradient_threshold, gradient_threshold)
            for g in gradients
        ]
    if gradient_norm_threshold > 0:
        gradients = [
            tf.clip_by_norm(g, gradient_norm_threshold) for g in gradients
        ]
    return gradients


def _construct_validation_summaries(metavalid_loss, metavalid_accuracy):
    tf.summary.scalar("metavalid_loss", metavalid_loss)
    tf.summary.scalar("metavalid_valid_accuracy", metavalid_accuracy)
    # The summaries are passed implicitly by TensorFlow.


def _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                  model_grads, model_vars):
    tf.summary.scalar("metatrain_loss", metatrain_loss)
    tf.summary.scalar("metatrain_valid_accuracy", metatrain_accuracy)
    for g, v in zip(model_grads, model_vars):
        histogram_name = v.name.split(":")[0]
        tf.summary.histogram(histogram_name, v)
        histogram_name = "gradient/{}".format(histogram_name)
        tf.summary.histogram(histogram_name, g)


def _construct_examples_batch(batch_size, split, num_classes,
                              num_tr_examples_per_class,
                              num_val_examples_per_class, db_path, sp_para=None):
    data_provider = data.DataProvider(split, config.get_data_config())
    data_provider.load_db(db_path)
    if sp_para:
        test_id, sp_bias, weights, k = sp_para
        data_provider.set_sp_paras(weights, sp_bias)
        data_provider.set_test_id(test_id, k)

    examples_batch = data_provider.get_batch(batch_size, num_classes,
                                             num_tr_examples_per_class,
                                             num_val_examples_per_class)
    return utils.unpack_data(examples_batch)


def _construct_loss_and_accuracy(inner_model, inputs, is_meta_training, test_batch=None):
    """Returns batched loss and accuracy of the model ran on the inputs."""
    call_fn = functools.partial(
        inner_model.__call__, is_meta_training=is_meta_training, test_data=test_batch)
    per_instance_loss, reg_loss, per_instance_accuracy = tf.map_fn(
        call_fn,
        inputs,
        dtype=(tf.float32, tf.float32, tf.float32),
        back_prop=is_meta_training)
    loss = tf.reduce_mean(per_instance_loss)
    accuracy = tf.reduce_mean(per_instance_accuracy)
    return loss, tf.reduce_mean(reg_loss), accuracy

def _construct_examples_batch_unlimited(batch_size, split, num_classes,
                              num_tr_examples_per_class,
                              num_val_examples_per_class):
    data_provider = data_unlimited.DataProvider(split, config.get_data_config())
    examples_batch = data_provider.get_batch(batch_size, num_classes,
                                             num_tr_examples_per_class,
                                             num_val_examples_per_class)
    return utils.unpack_data(examples_batch)


def construct_graph(outer_model_config, a, b, layers):
    """Constructs the optimization graph."""
    inner_model_config = config.get_inner_model_config()
    tf.logging.info("inner_model_config: {}".format(inner_model_config))
    num_classes = outer_model_config["num_classes"]
    maml = model.LeastSquareMeta(layers, a, num_classes, limited=False, l2_weight=b)


    num_tr_examples_per_class = outer_model_config["num_tr_examples_per_class"]
    metatrain_batch = _construct_examples_batch_unlimited(
        outer_model_config["metatrain_batch_size"], "train", num_classes,
        num_tr_examples_per_class,
        outer_model_config["num_val_examples_per_class"])
    metatrain_loss, _, metatrain_accuracy = _construct_loss_and_accuracy(
        maml, metatrain_batch, True)

    metatrain_gradients, metatrain_variables = maml.grads_and_vars(metatrain_loss)

    # Avoids NaNs in summaries.
    metatrain_loss = tf.cond(tf.is_nan(metatrain_loss),
                             lambda: tf.zeros_like(metatrain_loss),
                             lambda: metatrain_loss)

    metatrain_gradients = _clip_gradients(
        metatrain_gradients, outer_model_config["gradient_threshold"],
        outer_model_config["gradient_norm_threshold"])

    _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                  metatrain_gradients, metatrain_variables)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=outer_model_config["outer_lr"])
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(
        list(zip(metatrain_gradients, metatrain_variables)), global_step)

    data_config = config.get_data_config()
    tf.logging.info("data_config: {}".format(data_config))
    total_examples_per_class = data_config["total_examples_per_class"]
    metavalid_batch = _construct_examples_batch_unlimited(
        outer_model_config["metavalid_batch_size"], "val", num_classes,
        num_tr_examples_per_class,
        total_examples_per_class - num_tr_examples_per_class)
    metavalid_loss, _, metavalid_accuracy = _construct_loss_and_accuracy(
        maml, metavalid_batch, False)

    metatest_batch = _construct_examples_batch_unlimited(
        outer_model_config["metatest_batch_size"], "test", num_classes,
        num_tr_examples_per_class,
        total_examples_per_class - num_tr_examples_per_class)
    _, _, metatest_accuracy = _construct_loss_and_accuracy(
        maml, metatest_batch, False)
    _construct_validation_summaries(metavalid_loss, metavalid_accuracy)

    return (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
            metatest_accuracy)


def sp_construct_graph(lam, layers, outer_model_config, train_path, test_path, sp_para):
    """Constructs the optimization graph."""
    inner_model_config = config.get_inner_model_config()
    tf.logging.info("inner_model_config: {}".format(inner_model_config))
    num_classes = outer_model_config["num_classes"]

    leo = model.LeastSquareMeta(layers, lam, num_classes, limited=False, l2_weight=1e-6)

    test_data = data.DataProvider("test", config.get_data_config())
    test_data.load_db(test_path)

    num_tr_examples_per_class = outer_model_config["num_tr_examples_per_class"]

    metatrain_batch = _construct_examples_batch(
        outer_model_config["metatrain_batch_size"], "train", num_classes,
        num_tr_examples_per_class,
        outer_model_config["num_val_examples_per_class"], train_path, sp_para=sp_para)

    data_config = config.get_data_config()
    tf.logging.info("data_config: {}".format(data_config))
    total_examples_per_class = data_config["total_examples_per_class"]
    test_id, sp_bias, weights, k = sp_para
    metavalid_batch = _construct_examples_batch(
        1, "test", num_classes,
        num_tr_examples_per_class,
        total_examples_per_class - num_tr_examples_per_class, test_path, sp_para=(test_id, False, None, None))
    metavalid_loss, _, metavalid_accuracy = _construct_loss_and_accuracy(
        leo, metavalid_batch, False)

    metatrain_loss, reg_loss, metatrain_accuracy = _construct_loss_and_accuracy(
        leo, metatrain_batch, True, test_batch=metavalid_batch)

    metatrain_gradients, metatrain_variables = leo.grads_and_vars(metatrain_loss)

    metatrain_loss = tf.cond(tf.is_nan(metatrain_loss),
                             lambda: tf.zeros_like(metatrain_loss),
                             lambda: metatrain_loss)

    metatrain_gradients = _clip_gradients(
        metatrain_gradients, outer_model_config["gradient_threshold"],
        outer_model_config["gradient_norm_threshold"])

    _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                  metatrain_gradients, metatrain_variables)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=outer_model_config["outer_lr"])

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(
        list(zip(metatrain_gradients, metatrain_variables)), global_step)

    reset_optimizer_op = tf.variables_initializer(optimizer.variables())


    return (train_op, global_step, metatrain_loss, reg_loss, metavalid_accuracy, reset_optimizer_op)


def run_training_loop(checkpoint_path, a, b, layers):
    """Runs the training loop, either saving a checkpoint or evaluating it."""
    outer_model_config = config.get_outer_model_config()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("outer_model_config: {}".format(outer_model_config))


    (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
     metatest_accuracy) = construct_graph(outer_model_config, a, b, layers)

    num_steps_limit = outer_model_config["num_steps_limit"]
    best_metavalid_accuracy = 0.

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_path,
            save_summaries_steps=FLAGS.checkpoint_steps,
            log_step_count_steps=FLAGS.checkpoint_steps,
            save_checkpoint_steps=FLAGS.checkpoint_steps,
            # summary_dir=checkpoint_path,
            config=tf_config) as sess:
        if not FLAGS.evaluation_mode:
            global_step_ev = sess.run(global_step)
            while global_step_ev < num_steps_limit:
                if global_step_ev % FLAGS.checkpoint_steps == 0:
                    # Just after saving checkpoint, calculate accuracy 10 times and save
                    # the best checkpoint for early stopping.
                    metavalid_accuracy_ev = utils.evaluate_and_average(
                        sess, metavalid_accuracy, 10)
                    tf.logging.info("Step: {} meta-valid accuracy: {}".format(
                        global_step_ev, metavalid_accuracy_ev))

                    if metavalid_accuracy_ev > best_metavalid_accuracy:
                        utils.copy_checkpoint(checkpoint_path, global_step_ev,
                                              metavalid_accuracy_ev)
                        best_metavalid_accuracy = metavalid_accuracy_ev

                _, global_step_ev, metatrain_accuracy_ev = sess.run(
                    [train_op, global_step, metatrain_accuracy])
                if global_step_ev % (FLAGS.checkpoint_steps // 2) == 0:
                    tf.logging.info("Step: {} meta-train accuracy: {}".format(
                        global_step_ev, metatrain_accuracy_ev))
        else:
            assert not FLAGS.checkpoint_steps
            num_metatest_estimates = (
                    10000 // outer_model_config["metatest_batch_size"])

            test_accuracy = utils.evaluate_and_average(sess, metatest_accuracy,
                                                       num_metatest_estimates, has_std=True)
            return test_accuracy



def build_dist(feat, feat_2=None):
    if feat_2 is None:
        feat_2 = feat
    dist = np.sum(feat ** 2, axis=1, keepdims=True) + np.sum(feat_2 ** 2, axis=1)[np.newaxis, :] \
           - 2 * np.matmul(feat, feat_2.T)
    return dist


def build_db(checkpoint_path, db_name, sample_size, db_title=""):
    outer_model_config = config.get_outer_model_config()

    num_classes = outer_model_config["num_classes"]
    tr_size = outer_model_config["num_tr_examples_per_class"]

    if db_name == "test":
        val_size = 600 - tr_size
    else:
        val_size = outer_model_config["num_val_examples_per_class"]

    save_path = osp.join(checkpoint_path, "%s%s_%i_%i_%i" % (db_name, db_title, sample_size, tr_size, val_size))

    if not osp.exists(save_path):
        provider = data.DataProvider(db_name, config.get_data_config(), verbose=False)
        provider.create_db(sample_size, num_classes, tr_size, val_size)
        provider.save_db(save_path)

    return osp.basename(save_path)


def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_sig_matrix(db_path):
    db = unpickle(db_path)
    sigs = [e[0] for e in db]
    if isinstance(sigs[0], tuple):
        sigs = map(lambda x:x.astype(np.float64), map(np.stack, zip(*sigs)))
        return list(sigs)
    else:
        sigs = np.stack(sigs).astype(np.float64)
        return [sigs]

def merge_dist(dist_matrix, weights, func=None):
    ret = np.zeros_like(dist_matrix[0])
    for i in range(len(dist_matrix)):
        if func is None:
            ret += weights[i] * dist_matrix[i]
        else:
            ret += func(-weights[i] * dist_matrix[i])
    if func is None:
        del dist_matrix
        return np.exp(-ret)
    else:
        return ret / len(dist_matrix)


def compute_kernel(sigs, weights, func=None):
    n = sigs[0].shape[0]
    ret = np.zeros([n, n])

    for i in range(len(sigs)):
        if func is None:
            ret += weights[i] * build_dist(sigs[i])
        else:
            ret += func(-weights[i] * build_dist(sigs[i]))
    if func is None:
        return np.exp(-ret)
    else:
        return ret / len(sigs)


def solve_sp(checkpoint_path, supp_name, test_name, theta=20, lam=1e-8, extra_sigs=None, weights=(1, 0)):
    weight_name = supp_name + "|" + test_name
    weight_save = osp.join(checkpoint_path, weight_name)

    if not osp.exists(weight_save):
        test_save = osp.join(checkpoint_path, test_name)
        test_sigs = get_sig_matrix(test_save)

        train_save = osp.join(checkpoint_path, supp_name)
        train_sigs = get_sig_matrix(train_save)

        if extra_sigs is not None:
            tr_extra_sigs, test_extra_sigs = extra_sigs

            train_sigs.extend(tr_extra_sigs)
            test_sigs.extend(test_extra_sigs)

        train_test_dist = list(map(lambda x: build_dist(x[0], x[1]), zip(train_sigs, test_sigs)))
        B = merge_dist(train_test_dist, weights)
        del test_sigs, train_test_dist

        train_K = compute_kernel(train_sigs, weights)

        A = train_K + lam * np.identity(train_K.shape[0])

        factor = cho_factor(A)
        weights = cho_solve(factor, B)

        with open(weight_save, "wb") as f:
            pickle.dump(weights, f)

    return weight_save

def populate_db(db_title="", tr_size=1):
    FLAGS.num_tr_examples_per_class = tr_size
    if db_title == "":
        FLAGS.dataset_name = "miniImageNet"
    else:
        FLAGS.dataset_name = "tieredImageNet"
    train_path = build_db(save_root, "train", 30000, db_title=db_title)
    test_path = build_db(save_root, "test", 100, db_title=db_title)

    sp_weight_path = solve_sp(save_root, train_path, test_path, extra_sigs=None, theta=1, weights=(50, 0))
    return sp_weight_path


def sp_train_loop(lam, layers, checkpoint_path, test_id, train_path, test_path, weight_path, k, extra_steps, test_save=None):
    """Runs the training loop, either saving a checkpoint or evaluating it."""
    outer_model_config = config.get_outer_model_config()
    tf.logging.set_verbosity(tf.logging.ERROR)

    weights = unpickle(weight_path)
    (train_op, global_step, metatrain_accuracy, reg_loss, metavalid_accuracy, reset_optimizer_op) \
        = sp_construct_graph(lam, layers, outer_model_config, train_path, test_path, sp_para=(test_id, True, weights, k))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    if test_save is None:
        test_save = osp.join(checkpoint_path, "cold_%i" % test_id)
        save_freq = FLAGS.checkpoint_steps
    else:
        save_freq = None

    sp_acc = []

    train_loss = []

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=test_save,
            save_summaries_steps=save_freq,
            log_step_count_steps=save_freq,
            save_checkpoint_steps=save_freq,
            summary_dir=test_save,
            config=tf_config) as sess:
        sess.run(reset_optimizer_op)
        global_step_ev = sess.run(global_step)
        init_step = global_step_ev
        if extra_steps>0:
            base_acc = utils.evaluate_and_average(sess, metavalid_accuracy, 1)
            sp_acc.append(base_acc)
            num_steps_limit = global_step_ev + extra_steps
        else:
            num_steps_limit = outer_model_config["num_steps_limit"]
        while global_step_ev <= num_steps_limit:
            _, global_step_ev, metatrain_accuracy_ev, reg_loss_py = sess.run(
                [train_op, global_step, metatrain_accuracy, reg_loss])


            if global_step_ev % (FLAGS.checkpoint_steps // 2) == 0:
                tf.logging.info("Step: {} meta-train accuracy: {} reg_loss: {}".format(
                    global_step_ev, metatrain_accuracy_ev, reg_loss_py))

            if global_step_ev % FLAGS.checkpoint_steps == 0 and global_step_ev>init_step:
                # Just after saving checkpoint, calculate accuracy 10 times and save
                # the best checkpoint for early stopping.
                metavalid_accuracy_ev = utils.evaluate_and_average(
                    sess, metavalid_accuracy, 1)
                tf.logging.info("Step: {} meta-valid accuracy: {}".format(
                    global_step_ev, metavalid_accuracy_ev))

                sp_acc.append(metavalid_accuracy_ev)
                train_loss.append(metatrain_accuracy_ev)

    tf.reset_default_graph()
    return sp_acc, train_loss


def sp_warmstart_loop(lam, layers, old_path, save_root, test_id, extra_steps, train_path, test_path, weight_path, k):
    return sp_train_loop(lam, layers, save_root, test_id, train_path, test_path, weight_path, k, extra_steps, test_save=old_path)

def run_sp():
    warmstart_save = osp.abspath(osp.join(chkpoint_root, "lsmeta_5_0.100000_0.000001_640_640/best_checkpoint"))

    train_path = osp.join(save_root, "train_30000_5_15")
    test_path = osp.join(save_root, "test_100_5_595")
    weight_path = osp.join(save_root, "train_30000_5_15|test_100_5_595")
    k = 500
    extra_steps = 300

    FLAGS.outer_lr = 1e-4
    FLAGS.checkpoint_steps = 50
    FLAGS.dataset_name = "miniImageNet"
    FLAGS.num_tr_examples_per_class = 5

    accs = []

    lam = 0.1
    layers = (640, 640)

    for i in range(100):
        tmp, train_loss = sp_warmstart_loop(lam, layers, warmstart_save, chkpoint_root, i, extra_steps, train_path,
                                            test_path, weight_path, k)
        print("Task %i test accuracy every %d steps" % (i, FLAGS.checkpoint_steps), tmp)
        accs.append(tmp)

    print("Average accuracy for all tasks")
    print(np.mean(accs, axis=0))


def run_sp_tier():
    warmstart_save = osp.abspath(osp.join(chkpoint_root, "lsmetatier_5_0.100000_0.000001_640_640/best_checkpoint"))

    train_path = osp.join(save_root, "traintier_30000_5_15")
    test_path = osp.join(save_root, "testtier_100_5_595")
    weight_path = osp.join(save_root, "traintier_30000_5_15|testtier_100_5_595")
    k = 500
    extra_steps = 300

    FLAGS.outer_lr = 1e-4
    FLAGS.checkpoint_steps = 50
    FLAGS.dataset_name = "tieredImageNet"
    FLAGS.num_tr_examples_per_class = 5

    accs = []

    lam = 0.1
    layers = (640, 640)

    for i in range(100):
        tmp, train_loss = sp_warmstart_loop(lam, layers, warmstart_save, chkpoint_root, i, extra_steps, train_path,
                                            test_path, weight_path, k)
        print("Task %i test accuracy every %d steps" % (i, FLAGS.checkpoint_steps), tmp)
        accs.append(tmp)

    print("Average accuracy for all tasks")
    print(np.mean(accs, axis=0))


def run_sp_1shot():
    warmstart_save = osp.abspath(osp.join(chkpoint_root, "lsmeta_1_0.100000_0.000001_640_640/best_checkpoint"))

    tmp_root = osp.abspath(save_root)

    train_path = osp.join(tmp_root, "train_30000_1_15")
    test_path = osp.join(tmp_root, "test_100_1_599")
    weight_path = osp.join(tmp_root, "train_30000_1_15|test_100_1_599")
    k = 500
    extra_steps = 150

    FLAGS.outer_lr = 1e-4
    FLAGS.dataset_name = "miniImageNet"
    FLAGS.checkpoint_steps = 50
    FLAGS.num_tr_examples_per_class = 1

    lam = 0.1
    layers= (640, 640)

    accs = []

    for i in range(100):
        tmp, train_loss = sp_warmstart_loop(lam, layers, warmstart_save, chkpoint_root, i, extra_steps, train_path,
                                            test_path, weight_path, k)
        print("Task %i test accuracy every %d steps" % (i, FLAGS.checkpoint_steps), tmp)
        accs.append(tmp)

    print("Average accuracy for all tasks")
    print(np.mean(accs, axis=0))


def run_sp_1shot_tier():
    warmstart_save = osp.abspath(osp.join(chkpoint_root, "lsmetatier_1_0.100000_0.000001_640_640/best_checkpoint"))
    train_path = osp.join(save_root, "traintier_30000_1_15")
    test_path = osp.join(save_root, "testtier_100_1_599")
    weight_path = osp.join(save_root, "traintier_30000_1_15|testtier_100_1_599")
    k = 500
    extra_steps = 300

    FLAGS.outer_lr = 1e-4
    FLAGS.checkpoint_steps = 50
    FLAGS.num_tr_examples_per_class = 1
    FLAGS.dataset_name = "tieredImageNet"

    lam = 0.1
    layers= (640, 640)

    accs = []

    for i in range(100):
        tmp, train_loss = sp_warmstart_loop(lam, layers, warmstart_save, chkpoint_root, i, extra_steps, train_path, test_path, weight_path, k)
        print("Task %i test accuracy every %d steps" % (i, FLAGS.checkpoint_steps), tmp)
        accs.append(tmp)

    print("Average accuracy for all tasks")
    print(np.mean(accs, axis=0))

def norm_by_row(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def run_lsmeta_unlimited(a, b, layers=(256, 256), tr_size=5, db_title="", num=None):
    if db_title=="":
        FLAGS.dataset_name = "miniImageNet"
    else:
        FLAGS.dataset_name = "tieredImageNet"
    FLAGS.num_tr_examples_per_class = tr_size
    layer_str = "_".join(map(str, layers))
    FLAGS.checkpoint_steps = 500
    FLAGS.evaluation_mode = False
    FLAGS.num_steps_limit = 4000
    save_path = osp.abspath(osp.join(chkpoint_root, "lsmeta%s_%i_%f_%f_%s" % (db_title, tr_size, a, b, layer_str)))
    if num is not None:
        save_path += "_%i" % num
    run_training_loop(save_path, a, b, layers)

def test_lsmeta_unlimited(a, b, layers=(256, 256), tr_size=5, db_title=""):
    if db_title=="":
        FLAGS.dataset_name = "miniImageNet"
    else:
        FLAGS.dataset_name = "tieredImageNet"
    FLAGS.num_tr_examples_per_class = tr_size
    layer_str = "_".join(map(str, layers))
    save_path = osp.abspath(osp.join(chkpoint_root, "lsmeta%s_%i_%f_%f_%s/best_checkpoint" % (db_title, tr_size, a, b, layer_str)))
    FLAGS.checkpoint_steps = None
    FLAGS.evaluation_mode = True
    return run_training_loop(save_path, a, b, layers)

def main(argv):
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.makedirs(chkpoint_root, exist_ok=True)
    # db_title="" -> miniImageNet and db_title="tier" -> tieredImageNet.
    # tr_size=1 or 5 refers to 1 or 5 shots tasks.
    if argv[1] == "gen_db":
        print("Generating tasks and computing weights via structured prediction")
        populate_db(db_title="", tr_size=1)
    elif argv[1] == "uncon_meta":
        print("Learning warmstart parameters with unconditional meta-learning")
        run_lsmeta_unlimited(0.1, 1e-6, (640, 640), tr_size=1)
    elif argv[1] == "sp":
        print("Running structured prediction on 100 test tasks")
        run_sp_1shot()

    #structured prediction for 100 test tasks, starting from learned agnostic meta-parameters
    # run_sp()
    # run_sp_tier()
    # run_sp_1shot_tier()

    #test the agnostic meta-parameters learned above
    # print(test_lsmeta_unlimited(0.1, 1e-06, layers=(640, 640), tr_size=1, db_title=""))


if __name__ == "__main__":
    tf.app.run()
