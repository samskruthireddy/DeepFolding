from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell, LSTMCell, DropoutWrapper
import numpy as np
import data_utils
from tensorflow.python.platform import gfile
from layer_norm import LNLSTMCell, LNGRUCell, HairyLNGRUCell
import os

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', 'Directory for input data')
flags.DEFINE_string('data_file', 'ss_angle_full', 'Raw input file')
flags.DEFINE_integer('batch_size', 12, 'Batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('n_epochs', 100, 'Epochs')
flags.DEFINE_integer('limit_size', 16944, 'Maximum number of samples to load')
flags.DEFINE_integer('limit_len', 1700, 'Maximum length sequence to load')
flags.DEFINE_integer('report_interval', 1, 'Frequency of error reporting')
flags.DEFINE_boolean('load_only', False, 'Preprocess input file and quit')
flags.DEFINE_integer('checkpoint_epochs', 10, 'Checkpoint every n epochs')
flags.DEFINE_string('checkpoint_dir', 'ckpt', 'Directory for saving model checkpoints')
flags.DEFINE_integer('random_seed', 1, 'Random number seed')
flags.DEFINE_boolean('eval_only', False, 'Do evaluation only, not training')
flags.DEFINE_boolean('remove_angles', False, 'Set all angles to zero for evaluation')
flags.DEFINE_boolean('remove_aas', False, 'Remove AA sequence for evaluation (only)')
flags.DEFINE_float('angle_keep_ratio', 1.0, 'Parameter for angle dropout (both training and eval)')
flags.DEFINE_float('aa_keep_ratio', 1.0, 'Proportion of sequence data to discard (training and eval)')
flags.DEFINE_integer('initial_step', 0, 'Starting step count')
flags.DEFINE_boolean('print_predictions', True, 'Print out predictions')
flags.DEFINE_float('lstm_dropout_shallow', 0.0, 'Dropout rate on shallow RNN layer')
flags.DEFINE_float('lstm_dropout_deep', 0.0, 'Dropout rate on deep RNN layer')

dssp3_dict = {
  'G': 'H', 'H': 'H',
  'B': 'B', 'E': 'B',
  'I': 'C', 'T': 'C', 'C': 'C', 'S': 'C', '-': 'C', '.': 'C'
}

if FLAGS.random_seed is not None:
  print('Setting random seed to', FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

n_aa = len(data_utils.amino_acids)
n_hidden = 20
n_rnn1 = 20
n_rnn2 = 20

n_classes = len(data_utils.struct_cats)+1

print('Loading data')
data_file, names, lengths, aas, angles, cats, record_count, max_len, h5_indices = \
  data_utils.load_data(FLAGS.data_dir+'/'+FLAGS.data_file,
                       limit_size=FLAGS.limit_size, limit_len=FLAGS.limit_len)
if FLAGS.load_only is True:
  exit()
data = data_utils.DataSet(data_file, names, lengths, aas, angles, cats, record_count, max_len, h5_indices)
randIxs = np.random.permutation(data.record_count) #randomize batch order
n_train = len(randIxs) * 4 // 5
n_valid = len(randIxs) - n_train
print('n_train =', n_train, 'n_valid =', n_valid)
train_indices = randIxs[:n_train]
valid_indices = randIxs[n_train:]
#data.get_batch([1, 2, 3], angle_keep_ratio=FLAGS.angle_keep_ratio)

print('Defining graph')
graph = tf.Graph()

with graph.as_default():

  def bi_rnn(rnn_inputs, n_units, max_seq_len, dropout_rate=0.0, pack=True, name='ctc_rnn'):
    keep_prob = 1.0 - dropout_rate
    with tf.name_scope(name) as scope:
      normed = tf.contrib.layers.batch_norm(rnn_inputs, scope=scope)
      listed = tf.split(0, max_seq_len, tf.reshape(normed, [-1, n_units]), name='listed')
      rnn_fwd = LSTMCell(n_units, use_peepholes=True, state_is_tuple=True)
      rnn_back = LSTMCell(n_units, use_peepholes=True, state_is_tuple=True)
      if keep_prob != 1.0:
        rnn_fwd = DropoutWrapper(rnn_fwd, input_keep_prob=keep_prob)
        rnn_back = DropoutWrapper(rnn_fwd, input_keep_prob=keep_prob)
      rnn, _, _ = tf.nn.bidirectional_rnn(rnn_fwd, rnn_back, listed, dtype=tf.float32, scope=scope)
      with graph.name_scope('rnn_rs'):
        rnn_rs = [tf.reshape(t, [FLAGS.batch_size, 2, n_units]) for t in rnn]
      weights_out = tf.Variable(tf.truncated_normal([2, n_units],
                                                    stddev=np.sqrt(2.0 / (2*n_units))), name='weights_out')
      bias_out = tf.Variable(tf.zeros([n_units]), name='bias_out')
      with graph.name_scope('rnn_out'):
        rnn_out = [tf.reduce_sum(tf.mul(t, weights_out), reduction_indices=1) + bias_out for t in rnn_rs]
      if not pack:
        return rnn_out
      return tf.pack(rnn_out, name=scope)

  ####Graph input
  #batch_size = tf.placeholder(tf.int64, name='batch_size')
  batch_indices = tf.placeholder(tf.int64, name='batch_indices')
  batch_vals = tf.placeholder(tf.int32, name='batch_vals')
  batch_shape = tf.placeholder(tf.int64, name='batch_shape')
  targetY = tf.SparseTensor(batch_indices, batch_vals, batch_shape)
  seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')

  batch_aas = tf.placeholder(tf.float32, shape=[max_len, None, data_utils.n_aas], name='batch_aas')
  batch_angles = tf.placeholder(tf.float32, shape=[max_len, None, 2], name='batch_angles')
  inputs = tf.concat(2, [batch_aas, batch_angles], name='batch_inputs')
  inputs_rs = tf.reshape(inputs, [-1, data_utils.n_features], name='inputs_rs')

  input_hidden = tf.contrib.layers.fully_connected(inputs_rs, n_hidden,
                                                   activation_fn=tf.nn.elu, scope='input_hidden')

  rnn1 = bi_rnn(input_hidden, n_rnn1, max_len, dropout_rate=FLAGS.lstm_dropout_shallow, name='rnn1')
  rnn2 = bi_rnn(rnn1, n_rnn2, max_len, dropout_rate=FLAGS.lstm_dropout_deep, name='rnn2')

  pred_angles = tf.contrib.layers.fully_connected(rnn2, 2, activation_fn=tf.nn.elu)

  ####Optimizing
  logits = tf.contrib.layers.fully_connected(rnn2, n_classes, activation_fn=tf.nn.elu)
  logargmax = tf.argmax(logits, 2, name='logargmax')

  target_dense = tf.sparse_tensor_to_dense(targetY, name='target_dense')

  ####Evaluating
  cosine_distances = tf.contrib.losses.cosine_distance(batch_angles, pred_angles, 0, scope='cosine_distances')
  logitsMaxTest = tf.slice(logargmax, [0, 0], [seq_lengths[0], 1], name='logitsMaxTest')
  # decoded, _ = tf.nn.ctc_beam_search_decoder(logits3d, seq_lengths)
  # predictions = tf.to_int32(decoded[0], name='predictions')
  # editDistance = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=True)) / \
  #                tf.to_float(tf.size(targetY.values), name='editDistance')
  # tf.scalar_summary('editDistance', editDistance)
  loss = tf.reduce_mean(tf.add(tf.nn.ctc_loss(logits, targetY, seq_lengths,
                                              preprocess_collapse_repeated=False, ctc_merge_repeated=False),
                               tf.abs(cosine_distances)), name='loss')
  tf.scalar_summary('loss', loss)
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

  logits1d = tf.gather_nd(tf.transpose(logargmax), batch_indices, name='logits1d')
  accuracy = tf.contrib.metrics.accuracy(logits1d, tf.to_int64(batch_vals))
  tf.scalar_summary('accuracy', accuracy)

####Run session
with tf.Session(graph=graph) as session:
  print('Initializing')
  restored = False
  ii = FLAGS.initial_step
  global_step = 0
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.checkpoint_dir+'/train', session.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.checkpoint_dir+'/test', session.graph)
  saver = tf.train.Saver(tf.all_variables())
  tf.initialize_all_variables().run()
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from", str(ckpt.model_checkpoint_path))
    saver.restore(session, ckpt.model_checkpoint_path)
    restored = True
    dash_index = ckpt.model_checkpoint_path.index('-', None, -1)
    global_step = int(ckpt.model_checkpoint_path[-(len(ckpt.model_checkpoint_path) - dash_index - 1):]) + 1
    if ii == 0:
      ii = global_step * FLAGS.batch_size
  for epoch in range(global_step, FLAGS.n_epochs):
    print('Epoch', epoch+1, '...')
    epoch_indices = train_indices
    np.random.shuffle(epoch_indices)
    batchAccuracy = np.zeros((data.record_count // FLAGS.batch_size) + 1)
    batch_slices = zip(*(iter(epoch_indices),) * FLAGS.batch_size)
    for batch, batch_data_indices in enumerate(batch_slices):
      batchNames, batchSeqLengths, batchAas, batchAngles, batchTargetVals, \
      batchTargetShape, batchTargetIxs, \
       = data.get_batch(list(batch_data_indices),
                        angle_keep_ratio=FLAGS.angle_keep_ratio,
                        aa_keep_ratio=FLAGS.aa_keep_ratio)
      #assert(len(batchTargetVals) == np.sum(batchSeqLengths))
      if len(batchTargetVals) != np.sum(batchSeqLengths):
        print('n targets=', len(batchTargetVals), 'sum of lengths=', np.sum(batchSeqLengths))
        exit(1)
      feedDict = {batch_aas: batchAas, batch_angles: batchAngles, batch_indices: batchTargetIxs,
                  batch_vals: batchTargetVals,
                  batch_shape: batchTargetShape, seq_lengths: batchSeqLengths}
      lmt, l, _, acc, train_summary = \
        session.run([logitsMaxTest, loss, optimizer, accuracy,
                     merged], feed_dict=feedDict)
      train_writer.add_summary(train_summary, ii)
      ii += 1
      if (batch % FLAGS.report_interval) == 0:
        if FLAGS.print_predictions:
          first_pred = lmt[:,0]
          first_truth = (data.labels[batch_data_indices[0]])[:data.lengths[batch_data_indices[0]]]
          train_pred_string = ''.join([data_utils.cat_inv_dict[x] if x < len(data_utils.cat_inv_dict) else '.' for x in first_pred])
          train_label_string = ''.join([data_utils.cat_inv_dict[x] if x < len(data_utils.cat_inv_dict) else '.' for x in first_truth])
        else:
          train_pred_string = ''
          train_label_string = ''
        # Accuracy for validation data
        np.random.shuffle(valid_indices)
        dev_batch_indices = valid_indices[:FLAGS.batch_size]
        validNames, validSeqLengths, validAas, validAngles, validTargetVals, \
        validTargetShape, validTargetIxs = \
          data.get_batch(dev_batch_indices,
                         remove_angles=FLAGS.remove_angles,
                         remove_aas=FLAGS.remove_aas)
        feedDict = {batch_aas: validAas, batch_angles: validAngles,
                    batch_indices: validTargetIxs, batch_vals: validTargetVals,
                    batch_shape: validTargetShape, seq_lengths: validSeqLengths}
        dev_lmt, dev_acc, dev_summary = session.run([logitsMaxTest, accuracy, merged], feed_dict=feedDict)
        dev_example = dev_batch_indices[0]
        dev_first_pred = dev_lmt[:,0]
        dev_first_truth = (data.labels[dev_example])[:data.lengths[dev_example]]
        dev_pred_string = ''.join([data_utils.cat_inv_dict[x] if x < len(data_utils.cat_inv_dict) else '.' for x in dev_first_pred])
        dev_truth_string = ''.join([data_utils.cat_inv_dict[x] if x < len(data_utils.cat_inv_dict) else '.' for x in dev_first_truth])
        d3p = ''.join([dssp3_dict[x] for x in dev_pred_string])
        d3t = ''.join([dssp3_dict[x] for x in dev_truth_string])
        hbc_accuracy = (1.0 * sum([1 if p == t else 0 for p, t in zip(d3p, d3t)])) / len(d3p)
        d3h = d3t.count('H')
        d3b = d3t.count('B')
        d3c = d3t.count('C')
        if d3h > 0:
          h_correct = (1.0 * sum([1 if d3p[x] == 'H' and d3t[x] == 'H' else 0
                                  for x in range(len(d3p))])) / d3t.count('H')
        else:
          h_correct = 1.0
        if d3b > 0:
          b_correct = (1.0 * sum([1 if d3p[x] == 'B' and d3t[x] == 'B' else 0
                                  for x in range(len(d3p))])) / d3t.count('B')
        else:
          b_correct = 1.0
        if d3c > 0:
          c_correct = (1.0 * sum([1 if d3p[x] == 'C' and d3t[x] == 'C' else 0
                                  for x in range(len(d3p))])) / d3t.count('C')
        else:
          c_correct = 1.0
        q3_score = sum([h_correct, b_correct, c_correct]) / 3.0
        test_writer.add_summary(dev_summary, ii)
        test_writer.flush()
        print('Minibatch', batch, 'loss:', l, 'accuracy:', acc) #, 'tf accuracy:', tfacc)
        if FLAGS.print_predictions:
          print(train_pred_string)
          print(train_label_string)
        print('Dev batch accuracy:', dev_acc, 'example hbc accuracy:', hbc_accuracy, 'q3 score:', q3_score)
        if FLAGS.print_predictions:
          print(dev_pred_string)
          print(dev_truth_string)
      batchAccuracy[batch] = acc * len(batchSeqLengths)
      train_writer.flush()
    if FLAGS.checkpoint_epochs != 0 and (epoch % FLAGS.checkpoint_epochs) == 0:
      checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "ctc_secondary.ckpt")
      saver.save(session, checkpoint_path, global_step=epoch)
    epochAccuracy = batchAccuracy.sum() / record_count
    print('Epoch', epoch + 1, 'accuracy:', epochAccuracy)
