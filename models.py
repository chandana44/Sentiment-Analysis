# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *
import datetime


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def processdata(exs, seq_max_len):
    mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in exs])
    seq_lens = np.array([len(ex.indexed_words) for ex in exs])
    labels_arr = np.array([ex.label for ex in exs])
    return mat, seq_lens, labels_arr

# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    word_vector_dimension = 300
    test_results = []

    train_mat, train_seq_lens, train_labels_arr = processdata(train_exs, seq_max_len)
    valid_mat, valid_seq_lens, valid_labels_arr = processdata(dev_exs, seq_max_len)
    test_mat, test_seq_lens, test_labels_arr = processdata(test_exs, seq_max_len)

    # MAKE THE DATA
    # Define some constants
    feat_vec_size = word_vector_dimension
    embedding_size1 = 150
    embedding_size2 = 75
    embedding_size3 = 25
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    fx = tf.placeholder(tf.float32, feat_vec_size)
    V1 = tf.get_variable("V1", [embedding_size1, feat_vec_size],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    z1 = tf.sigmoid(tf.tensordot(V1, fx, 1))
    V2 = tf.get_variable("V2", [embedding_size2, embedding_size1],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    z2 = tf.sigmoid(tf.tensordot(V2, z1, 1))
    V3 = tf.get_variable("V3", [embedding_size3, embedding_size2],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    z3 = tf.sigmoid(tf.tensordot(V3, z2, 1))
    W = tf.get_variable("W", [num_classes, embedding_size3])
    probs = tf.nn.softmax(tf.tensordot(W, z3, 1))
    # prediction
    one_best = tf.argmax(probs)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, 1)
    # one hot vectors for labels
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))

    # TRAINING ALGORITHM CUSTOMIZATION
    decay_steps = 8000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    initial_learning_rate = 0.001
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_mat)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict={
                    fx: np.mean(np.array([word_vectors.vectors[int(wordindex)] for wordindex in train_mat[ex_idx]]),0),
                    label: np.array([train_labels_arr[ex_idx]])})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        # Evaluate on the train set
        train_correct = 0
        for ex_idx in xrange(0, len(train_mat)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z3],
                                                                                  feed_dict={fx: np.mean(np.array([word_vectors.vectors[int(wordindex)] for wordindex in train_mat[ex_idx]]),0)})
            if (train_labels_arr[ex_idx] == pred_this_instance):
                train_correct += 1
        print repr(train_correct) + "/" + repr(len(train_labels_arr)) + " correct after training"

        # Evaluate on the dev set
        valid_correct = 0
        for ex_idx in xrange(0, len(valid_mat)):
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z3],
                                                                                  feed_dict={fx: np.mean(np.array([word_vectors.vectors[int(wordindex)] for wordindex in valid_mat[ex_idx]]),0)})
            if (valid_labels_arr[ex_idx] == pred_this_instance):
                valid_correct += 1
        print repr(valid_correct) + "/" + repr(len(valid_labels_arr)) + " correct for dev"

        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_mat)):
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z3],
                                                                                  feed_dict={fx: np.mean(np.array([word_vectors.vectors[int(wordindex)] for wordindex in test_mat[ex_idx]]),0)})
            test_results.append(SentimentExample(test_exs[ex_idx].indexed_words, pred_this_instance))
            if (test_labels_arr[ex_idx] == pred_this_instance):
                test_correct += 1
        print repr(test_correct) + "/" + repr(len(test_labels_arr)) + " correct for test"

    return test_results

def getDataBatch(data_mat, labels_arr, batchSize, maxSeqLength):
    labels = np.zeros([batchSize, 2])
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        arr[i]=data_mat[i]
        labels[i][labels_arr[i]] = 1
    return arr, labels


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    bilstm = False

    maxSeqLength = 60
    numDimensions = 300
    batchSize = 1
    lstmUnits = 32
    numClasses = 2
    iterations = 7

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    train_mat, train_seq_lens, train_labels_arr = processdata(train_exs, maxSeqLength)
    valid_mat, valid_seq_lens, valid_labels_arr = processdata(dev_exs, maxSeqLength)
    test_mat, test_seq_lens, test_labels_arr = processdata(test_exs, maxSeqLength)

    train_data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    train_data = tf.nn.embedding_lookup(word_vectors.vectors, input_data)
    train_data = tf.cast(train_data, tf.float32)

    if(not bilstm):
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, train_data, dtype=tf.float32)
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    else:
        forward_cell = tf.contrib.rnn.LSTMCell(lstmUnits)
        forward_cell = tf.contrib.rnn.DropoutWrapper(cell=forward_cell, output_keep_prob=0.75)
        backward_cell = tf.contrib.rnn.LSTMCell(lstmUnits)
        backward_cell = tf.contrib.rnn.DropoutWrapper(cell=backward_cell, output_keep_prob=0.75)
        value, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                 train_data, dtype=tf.float32)
        value = tf.concat(value, 2)
        weight = tf.Variable(tf.truncated_normal([2*lstmUnits, numClasses]))


    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    predictedValue = tf.argmax(prediction, 1)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    #optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(iterations):
        train_correct = 0
        loss_this_iter = 0
        for ex_idx in xrange(0, len(train_mat)):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = getDataBatch(train_mat[ex_idx:ex_idx+1], train_labels_arr[ex_idx:ex_idx+1], batchSize, maxSeqLength)
            _, predict, lossval = sess.run([optimizer, predictedValue, loss], {input_data: nextBatch, labels: nextBatchLabels})
            if (train_labels_arr[ex_idx] == predict):
                train_correct += 1
            loss_this_iter += lossval
            if(ex_idx%1000 == 0):
                print str(ex_idx) + '/' + str(len(train_mat))

        print str(i) + '/' + str(iterations) + ' complete'
        print repr(train_correct) + "/" + repr(len(train_labels_arr)) + " correct for train" + '   loss: ' + str(loss_this_iter)

        summary = tf.Summary()
        summary.value.add(tag='Training Loss', simple_value=loss_this_iter)
        writer.add_summary(summary, i)

    writer.close()

    # Evaluate on the valid set
    valid_correct = 0
    for ex_idx in xrange(0, len(valid_mat)):
        nextBatch, nextBatchLabels = getDataBatch(valid_mat[ex_idx:ex_idx + 1], valid_labels_arr[ex_idx:ex_idx + 1],
                                                  batchSize,
                                                  maxSeqLength)
        predict = sess.run(predictedValue, {input_data: nextBatch})
        if (valid_labels_arr[ex_idx] == predict):
            valid_correct += 1
    print repr(valid_correct) + "/" + repr(len(valid_labels_arr)) + " correct for dev"

    # Evaluate on the test set
    test_correct = 0
    test_results = []
    for ex_idx in xrange(0, len(test_mat)):
        nextBatch, nextBatchLabels = getDataBatch(test_mat[ex_idx:ex_idx+1], test_labels_arr[ex_idx:ex_idx+1], batchSize, maxSeqLength)
        predict = sess.run(predictedValue, {input_data: nextBatch})
        test_results.append(SentimentExample(test_exs[ex_idx].indexed_words, predict))
        if (test_labels_arr[ex_idx] == predict):
                test_correct += 1
    print repr(test_correct) + "/" + repr(len(valid_labels_arr)) + " correct for test"

    return test_results
