# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""To perform inference on test set given a trained model."""
from __future__ import print_function

import codecs
import time

import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils
from .utils import nmt_utils

__all__ = ["load_data", "inference",
           "single_worker_inference", "multi_worker_inference"]


def _decode_inference_indices(model, sess, output_infer,
                              output_infer_summary_prefix,
                              inference_indices,
                              tgt_eos,
                              subword_option):
  """Decoding only a specific set of sentences."""
  utils.print_out("  decoding to output %s , num sents %d." %
                  (output_infer, len(inference_indices)))
  start_time = time.time()
  with codecs.getwriter("utf-8")(
      tf.gfile.GFile(output_infer, mode="wb")) as trans_f:
    trans_f.write("")  # Write empty string to ensure file is created.
    for decode_id in inference_indices:
      nmt_outputs, infer_summary = model.decode(sess)

      # get text translation
      assert nmt_outputs.shape[0] == 1
      translation = nmt_utils.get_translation(
          nmt_outputs,
          sent_id=0,
          tgt_eos=tgt_eos,
          subword_option=subword_option)

      if infer_summary is not None:  # Attention models
        image_file = output_infer_summary_prefix + str(decode_id) + ".png"
        utils.print_out("  save attention image to %s*" % image_file)
        image_summ = tf.Summary()
        image_summ.ParseFromString(infer_summary)
        with tf.gfile.GFile(image_file, mode="w") as img_f:
          img_f.write(image_summ.value[0].image.encoded_image_string)

      trans_f.write("%s\n" % translation)
      utils.print_out(translation + b"\n")
  utils.print_time("  done", start_time)


def load_data(inference_input_file, hparams=None):
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  if hparams and hparams.inference_indices:
    inference_data = [inference_data[i] for i in hparams.inference_indices]

  return inference_data


def inference(ckpt,
              inference_input_file,
              inference_output_file,
              hparams,
              num_workers=1,
              jobid=0,
              scope=None):
  """Perform translation."""
  if hparams.inference_indices:
    assert num_workers == 1

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  if num_workers == 1:
    single_worker_inference(
        infer_model,
        ckpt,
        inference_input_file,
        inference_output_file,
        hparams)
  else:
    multi_worker_inference(
        infer_model,
        ckpt,
        inference_input_file,
        inference_output_file,
        hparams,
        num_workers=num_workers,
        jobid=jobid)


def vectorisation(ckpt,
                  vectorisation_input_file,
                  vectorisation_output_file,
                  hparams,
                  num_workers=1,
                  jobid=0,
                  scope=None):
  """Perform vectorisation - convert random length samples to fixed length context vectors."""
  if hparams.inference_indices:
    assert num_workers == 1

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")
  vectorisation_model = model_helper.create_vector_model(model_creator, hparams, scope)

  if num_workers == 1:
    single_worker_vectorisation(
        vectorisation_model,
        ckpt,
        vectorisation_input_file,
        vectorisation_output_file,
        hparams)
  else:
    raise NotImplementedError("Multi worker vectorisation not supported.")
    # multi_worker_inference(
    #     vectorisation_model,
    #     ckpt,
    #     inference_input_file,
    #     inference_output_file,
    #     hparams,
    #     num_workers=num_workers,
    #     jobid=jobid)


def single_worker_inference(infer_model,
                            ckpt,
                            inference_input_file,
                            inference_output_file,
                            hparams):
  """Inference with a single worker."""
  output_infer = inference_output_file

  # Read data
  infer_data = load_data(inference_input_file, hparams)

  with tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto()) as sess:
    loaded_infer_model = model_helper.load_model(
        infer_model.model, ckpt, sess, "infer", hparams)
    sess.run(
        infer_model.iterator.initializer,
        feed_dict={
            infer_model.src_placeholder: infer_data,
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        })
    # Decode
    utils.print_out("# Start decoding")
    if hparams.inference_indices:
      _decode_inference_indices(
          loaded_infer_model,
          sess,
          output_infer=output_infer,
          output_infer_summary_prefix=output_infer,
          inference_indices=hparams.inference_indices,
          tgt_eos=hparams.eos,
          subword_option=hparams.subword_option)
    else:
      nmt_utils.decode_and_evaluate(
          "infer",
          loaded_infer_model,
          sess,
          output_infer,
          ref_file=None,
          metrics=hparams.metrics,
          subword_option=hparams.subword_option,
          beam_width=hparams.beam_width,
          tgt_eos=hparams.eos,
          num_translations_per_input=hparams.num_translations_per_input)


def single_worker_vectorisation(vector_model,
                                ckpt,
                                vectorisation_input_file,
                                vectorisation_output_file,
                                hparams):
  """Inference with a single worker."""
  output_vector = vectorisation_output_file

  # Read data
  vectorisation_data = load_data(vectorisation_input_file, hparams)

  with tf.Session(
      graph=vector_model.graph, config=utils.get_config_proto()) as sess:
    feed_dict={
      vector_model.src_placeholder: vectorisation_data,
      vector_model.batch_size_placeholder: hparams.infer_batch_size,
      vector_model.src_vocab_file_name_ph: hparams.src_vocab_file
    }
    loaded_vector_model = model_helper.load_model(
        vector_model.model, ckpt, sess, "infer", feed_dict)
    sess.run(
        vector_model.iterator.initializer,
        feed_dict=feed_dict)
    # Decode
    utils.print_out("# Start encoding")
    if hparams.inference_indices:
      raise NotImplementedError("Vectorisation inference_indices not supported.")
      # _decode_inference_indices(
      #     loaded_vector_model,
      #     sess,
      #     output_infer=output_infer,
      #     output_infer_summary_prefix=output_infer,
      #     inference_indices=hparams.inference_indices,
      #     tgt_eos=hparams.eos,
      #     subword_option=hparams.subword_option)
    else:
      vector_summary = nmt_utils.vectorize(loaded_vector_model,
                                           sess,
                                           output_vector,
                                           beam_width=hparams.beam_width,
                                           num_vectors_per_input=hparams.num_translations_per_input)

    save_vectorizer(ckpt, sess, vector_model, vector_summary)


def save_vectorizer(ckpt, sess, vector_model, vector_summary):
  # ===========================================================
  #  The saving of the model / checkpoints / logs, etc.
  # ===========================================================
  ckpt_dirname = os.path.join(os.path.dirname(ckpt), "vectorizer")

  # so I can see the vectorization-only graph in TensorBoard
  summary_writer = tf.summary.FileWriter(ckpt_dirname, vector_model.graph)
  summary_writer.add_summary(vector_summary)
  summary_writer.flush()

  # Save vectorization model for reuse later (Java code)
  vector_model.model.saver.save(sess, os.path.join(ckpt_dirname, "vectorizer.ckpt"))

  model_export_dir = os.path.join(ckpt_dirname, "model_export")

  builder = tf.saved_model.builder.SavedModelBuilder(model_export_dir)
  builder.add_meta_graph_and_variables(sess,
                                       ["vectorize"],
                                       clear_devices=True)
  builder.save()


def multi_worker_inference(infer_model,
                           ckpt,
                           inference_input_file,
                           inference_output_file,
                           hparams,
                           num_workers,
                           jobid):
  """Inference using multiple workers."""
  assert num_workers > 1

  final_output_infer = inference_output_file
  output_infer = "%s_%d" % (inference_output_file, jobid)
  output_infer_done = "%s_done_%d" % (inference_output_file, jobid)

  # Read data
  infer_data = load_data(inference_input_file, hparams)

  # Split data to multiple workers
  total_load = len(infer_data)
  load_per_worker = int((total_load - 1) / num_workers) + 1
  start_position = jobid * load_per_worker
  end_position = min(start_position + load_per_worker, total_load)
  infer_data = infer_data[start_position:end_position]

  with tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto()) as sess:
    loaded_infer_model = model_helper.load_model(
        infer_model.model, ckpt, sess, "infer")
    sess.run(infer_model.iterator.initializer,
             {
                 infer_model.src_placeholder: infer_data,
                 infer_model.batch_size_placeholder: hparams.infer_batch_size
             })
    # Decode
    utils.print_out("# Start decoding")
    nmt_utils.decode_and_evaluate(
        "infer",
        loaded_infer_model,
        sess,
        output_infer,
        ref_file=None,
        metrics=hparams.metrics,
        subword_option=hparams.subword_option,
        beam_width=hparams.beam_width,
        tgt_eos=hparams.eos,
        num_translations_per_input=hparams.num_translations_per_input)

    # Change file name to indicate the file writing is completed.
    tf.gfile.Rename(output_infer, output_infer_done, overwrite=True)

    # Job 0 is responsible for the clean up.
    if jobid != 0: return

    # Now write all translations
    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(final_output_infer, mode="wb")) as final_f:
      for worker_id in range(num_workers):
        worker_infer_done = "%s_done_%d" % (inference_output_file, worker_id)
        while not tf.gfile.Exists(worker_infer_done):
          utils.print_out("  waitting job %d to complete." % worker_id)
          time.sleep(10)

        with codecs.getreader("utf-8")(
            tf.gfile.GFile(worker_infer_done, mode="rb")) as f:
          for translation in f:
            final_f.write("%s" % translation)

      for worker_id in range(num_workers):
        worker_infer_done = "%s_done_%d" % (inference_output_file, worker_id)
        tf.gfile.Remove(worker_infer_done)
