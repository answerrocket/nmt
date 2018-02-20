import argparse
from fnmatch import fnmatch

import tensorflow as tf
import numpy as np
import sys
import os
import random
import collections

arguments = None
log_dir = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.register("type", "int_list", lambda v: [int(s) for s in v.split(',')])

  parser.add_argument("--mode", type=str, default="TRAIN",
                      help="TRAIN | INFER")

  # network
  parser.add_argument("--hidden_layer_sizes", type="int_list", default=[256],
                      help="Array of ints describing hidden layer topology.  "
                           "A hidden layer of the given size is created for every int in the list.")

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))

  # training
  parser.add_argument("--num_epochs", type=int, default=100,
                      help="Number of epochs (passes over training data) to run during training")
  parser.add_argument("--epoch_print_interval", type=int, default=10,
                      help="Number of epochs per print of training status.")
  parser.add_argument("--epoch_ckpt_interval", type=int, default=10,
                      help="Number of epochs per full checkpoint.")
  parser.add_argument("--epoch_tb_log_interval", type=int, default=1,
                      help="Number of epochs TensorBoard update of training status.")
  parser.add_argument("--num_keep_ckpts", type=int, default=5,
                      help="Max number of checkpoints to keep.")
  parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
  parser.add_argument("--learning_rate", type=float, default=1.0,
                      help="Learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument("--ckpt", type=str, default=None,
                      help="Checkpoint to load / start from")

  # data
  parser.add_argument("--src", type=str, default=None,
                      help="Source suffix, e.g., en.")
  parser.add_argument("--tgt", type=str, default=None,
                      help="Target suffix, e.g., de.")
  parser.add_argument("--train_prefix", type=str, default=None,
                      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--dev_prefix", type=str, default=None,
                      help="Dev prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--test_prefix", type=str, default=None,
                      help="Test prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--out_dir", type=str, default=None,
                      help="Store log/model files.")
  parser.add_argument("--raw_source", type=str, default=None,
                      help="Source input prior to any preprocessing that may have been done (this is for final reporting).")
  parser.add_argument("--input_width", type=int, default=512,
                      help="Number of floats in each input vector")
  parser.add_argument("--output_width", type=int, default=2,
                      help="Number of classes to discern")
  parser.add_argument("--infer_input_file", type=str, default=None,
                      help="File of vectors to run classification on.")

  # parser.add_argument("--dropout", type=float, default=0.2,
  #                     help="Dropout rate (not keep_prob)")

  # Inference
  parser.add_argument("--inference_input_file", type=str, default=None,
                      help="Set to the text to decode.")
  parser.add_argument("--inference_output_file", type=str, default=None,
                      help="Output file to store decoding results.")

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape, name=None):
  """ Weight initialization """
  weights = tf.random_normal(shape, stddev=0.1, name=(None if name is None else "random_normal_" + name))
  return tf.Variable(weights,
                     # name=(None if name is None else "weights_" + name)
                     )


def forwardprop(X, w_1, w_2):
  """
  Forward-propagation.
  IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
  """
  h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
  yhat = tf.matmul(h, w_2)  # The \varphi function
  return yhat


def make_file_name(prefix, extension=None):
  if extension:
    return prefix + "." + extension
  else:
    return prefix


def prepend_bias_column(matrix2d):
  n, m = matrix2d.shape
  result = np.ones((n, m + 1))
  result[:, 1:] = matrix2d
  return result


def convert_to_one_hot(index_vector):
  num_labels = len(np.unique(index_vector))
  result = np.eye(num_labels)[index_vector]  # One liner trick!
  return result


def load_training_corpus():

  train_src = load_source_data(arguments.train_prefix, arguments.src)
  test_src = load_source_data(arguments.test_prefix, arguments.src)

  train_tgt = load_target_data(arguments.train_prefix, arguments.tgt)
  test_tgt = load_target_data(arguments.test_prefix, arguments.tgt)

  return train_src, test_src, train_tgt, test_tgt


def load_target_data(prefix, extension=None):
  tgt = load_ints(prefix, extension)
  tgt = convert_to_one_hot(tgt)
  return tgt


def load_source_data(prefix, extension=None):
  src = load_floats(prefix, extension)
  src = prepend_bias_column(src)
  return src


def load_floats(prefix, extension=None):
  return np.loadtxt(make_file_name(prefix, extension), delimiter=",")


def load_ints(prefix, extension=None):
  return np.loadtxt(make_file_name(prefix, extension), delimiter=",", dtype=int)


def load_text_lines(file_name):
  with open(file_name, "r") as f:
    return [s.rstrip() for s in f.readlines()]


class FeedForwardModel(
  collections.namedtuple("FeedForwardModel",
                         ("graph",
                          "input_ph",
                          "output_ph",
                          "num_hidden_layers",
                          "weights",
                          "hidden_layers",
                          "yhat",
                          "predict",
                          "cost",
                          "updates",
                          "saver",
                          "summary_writer",
                          "train_summary",
                          "classifier_summary"))):
  pass


def build_model(source_size, target_size):

  graph = tf.get_default_graph()

  # Symbols
  input_ph = tf.placeholder("float", shape=[None, source_size], name="VectorInput_ph")
  output_ph = tf.placeholder("float", shape=[None, target_size], name="ClassificationTrainingLabel_ph")

  # Initializer
  tf.get_variable_scope().set_initializer(get_initializer())

  num_layers = len(arguments.hidden_layer_sizes)

  assert num_layers > 0, "Must have at least one layer"  # technically you could have 0, but that won't work here anyway

  all_layer_sizes = [source_size] + arguments.hidden_layer_sizes + [target_size]

  weights = []
  hidden_layers = []
  previous_stage = input_ph

  # Weight initializations
  for i in range(len(all_layer_sizes) - 1):
    w = init_weights((all_layer_sizes[i], all_layer_sizes[i + 1]), name="layer_"+str(i))
    weights.append(w)
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.matmul(previous_stage, w)

    if i < num_layers: # last layer is the output layer - no sigmoid
      h = tf.nn.sigmoid(h)  # The \sigma function
      hidden_layers.append(h)

    previous_stage = h

  yhat = previous_stage
  predict = tf.argmax(yhat, axis=1, name="Prediction")

  # Backward propagation
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_ph, logits=yhat))
  optimizer = tf.train.GradientDescentOptimizer(arguments.learning_rate)
  updates = optimizer.minimize(cost)

  accuracy = tf.div(tf.count_nonzero(tf.equal(tf.cast(tf.argmax(output_ph, axis=1), tf.int64), predict, name="equal_out_predict"),
                                     axis=0,
                                     name="count_correct",
                                     dtype=tf.float32),
                    tf.cast(tf.shape(predict, name="batch_size")[0], tf.float32),
                    name="accuracy_calc")
  # Saver
  saver = tf.train.Saver(
    tf.global_variables(), max_to_keep=arguments.num_keep_ckpts)

  # train_summary = tf.summary.scalar("cost", cost)

  train_summary = tf.summary.merge([tf.summary.scalar("cost", cost),
                                    tf.summary.scalar("accuracy", accuracy)])

  classifier_summary = tf.summary.scalar("the_secret_of_the_universe", tf.constant(42))

  summary_writer = tf.summary.FileWriter(choose_unique_log_dir(), graph, flush_secs=20)

  return FeedForwardModel(
    graph=graph,
    input_ph=input_ph,
    output_ph=output_ph,
    num_hidden_layers=num_layers,
    weights=weights,
    hidden_layers=hidden_layers,
    yhat=yhat,
    predict=predict,
    cost=cost,
    updates=updates,
    saver=saver,
    summary_writer=summary_writer,
    train_summary=train_summary,
    classifier_summary=classifier_summary
  )


LOG_NAME_STUB = "train_log_"


def choose_unique_log_dir():
  global log_dir

  if log_dir is None:
    log_dir = LOG_NAME_STUB + str(find_max_dir_num() + 1)

  return log_dir


def latest_output_log_dir():
  return os.path.join(arguments.out_dir, LOG_NAME_STUB + str(find_max_dir_num()))


def find_max_dir_num():
  found_log_dirs = [d for d in os.listdir(arguments.out_dir)
                    if os.path.isdir(os.path.join(arguments.out_dir, d))
                    and fnmatch(d, LOG_NAME_STUB + "*")]

  return max([0] +
             [int(l[len(LOG_NAME_STUB):])
              for l in found_log_dirs])


def evaluate_model(model, sess, source, target):
  test_accuracy = np.mean(np.argmax(target, axis=1) ==
                          sess.run(model.predict,
                                   feed_dict={model.input_ph: source,
                                              model.output_ph: target}))
  test_cost = sess.run(model.cost,
                       feed_dict={model.input_ph: source,
                                  model.output_ph: target})

  return test_accuracy, test_cost


def get_initializer():
  init_op = arguments.init_op
  seed = None
  init_weight = arguments.init_weight

  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(
      -init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(
      seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(
      seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


def dump_output(model, sess, source, target, raw_source=None, raw_target=None):

  # raw_source = raw_source or [str(s) for s in source]
  # raw_target = raw_target or [str(t) for t in target]

  dir = choose_unique_log_dir()
  if not os.path.exists(dir):
    os.makedirs(dir)

  file_path = os.path.join(dir, "final_test_output.csv")

  with open(file_path, "w") as fp:
    fp.write("Question, TargetClass, yHat, Predicted Class\n")

    feed_dict = {model.input_ph: source}
    predictions, yhat = sess.run([model.predict, model.yhat], feed_dict=feed_dict)
    # yhat = sess.run(model.yhat, feed_dict=feed_dict)
    yhat = [":".join(["%.2f" % (f) for f in y]) for y in yhat.tolist()]

    for data in zip(raw_source, raw_target, yhat, predictions):
      fp.write("%s,%.2f,%s,%.2f\n" % data)


def save_checkpoint(model, sess, global_step, use_log_dir=False):
  dir_to_use = choose_unique_log_dir() if use_log_dir else arguments.out_dir
  model.saver.save(sess,
                   os.path.join(dir_to_use, "translate.ckpt"),
                   global_step=global_step)


def train_classifier():
  train_src, test_src, train_tgt, test_tgt = load_training_corpus()

  # Layer's sizes
  src_size = train_src.shape[1]
  tgt_size = train_tgt.shape[1]

  model = build_model(src_size, tgt_size)

  # Run SGD
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  # ================================================
  epoch = 0
  epoch_len = len(train_src)
  global_step = 0

  for epoch in range(arguments.num_epochs):
    # Train with each example
    for i in range(epoch_len):
      next_i = i + 1
      feed_dict = {model.input_ph: train_src[i: next_i], model.output_ph: train_tgt[i: next_i]}
      sess.run(model.updates, feed_dict=feed_dict)

    evaluation = None

    global_step = (epoch + 1) * epoch_len

    if epoch % arguments.epoch_tb_log_interval == 0:
      last_summary = sess.run(model.train_summary,
                              feed_dict={model.input_ph: test_src,
                                         model.output_ph: test_tgt})
      model.summary_writer.add_summary(last_summary, global_step)

    if epoch % arguments.epoch_ckpt_interval == 0 and epoch > 0:
      save_checkpoint(model, sess, global_step)

    if epoch % arguments.epoch_print_interval == 0:
      test_accuracy, test_cost = evaluation = evaluate_model(model, sess, test_src, test_tgt) if evaluation is None else evaluation
      print("Epoch = %d, test accuracy = %.2f%%, test cost = %.7f"
            % (epoch + 1, 100. * float(test_accuracy), test_cost))

  # ================================================
  model.summary_writer.flush()

  print("\n\nFinal Results:\n")

  test_accuracy, test_cost = evaluate_model(model, sess, test_src, test_tgt)
  print("Epoch = %d, test accuracy = %.2f%%, test cost = %.7f"
        % (epoch + 1, 100. * float(test_accuracy), test_cost))

  if arguments.out_dir:
    dump_output(model,
                sess,
                test_src,
                test_tgt,
                load_text_lines(arguments.raw_source),
                load_ints(arguments.test_prefix, arguments.tgt))
    save_checkpoint(model, sess, global_step, use_log_dir=True)

  sess.close()


def should_run_inference():
  return arguments.mode == "INFER"


def get_checkpoint():
  ckpt = arguments.ckpt
  if not ckpt:
    ckpt = tf.train.latest_checkpoint(arguments.out_dir)
  if not ckpt:
    ckpt = tf.train.latest_checkpoint(latest_output_log_dir())

  return ckpt


def run_classifier():
  ckpt = get_checkpoint()

  assert ckpt is not None

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  model = build_model(arguments.input_width + 1, arguments.output_width)

  model.saver.restore(sess, ckpt)
  sess.run(tf.tables_initializer())

  source_data = load_source_data(arguments.infer_input_file)

  classifications, classifier_summary = sess.run([model.predict,
                                                  model.classifier_summary],
                                                 feed_dict={model.input_ph: source_data})

  # TODO: write output to a file or something....
  print("\n\n" + str(classifications) + "\n")

  save_classifier(ckpt, sess, model, classifier_summary)


def save_classifier(ckpt, sess, model, classifier_summary):
  # ===========================================================
  #  The saving of the model / checkpoints / logs, etc.
  # ===========================================================
  ckpt_dirname = os.path.join(os.path.dirname(ckpt), "classifier")

  # so I can see the classification-only graph in TensorBoard
  summary_writer = tf.summary.FileWriter(ckpt_dirname, model.graph)
  summary_writer.add_summary(classifier_summary)
  summary_writer.flush()

  # Save vectorization model for reuse later (Java code)
  model.saver.save(sess, os.path.join(ckpt_dirname, "classifier.ckpt"))

  model_export_dir = os.path.join(ckpt_dirname, "model_export")

  builder = tf.saved_model.builder.SavedModelBuilder(model_export_dir)
  builder.add_meta_graph_and_variables(sess,
                                       ["classify"],
                                       clear_devices=True)
  builder.save()


def main(unused_argv):
  if should_run_inference():
    run_classifier()
  else:
    train_classifier()


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  add_arguments(arg_parser)
  arguments, unparsed = arg_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)