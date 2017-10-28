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
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in exs])

    # Also store the sequence lengths -- this could be useful for training LSTMs
    seq_lens = np.array([len(ex.indexed_words) for ex in exs])
    # Labels
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
    # Let's use 10 hidden units
    embedding_size1 = feat_vec_size / 3
    embedding_size2 = embedding_size1 / 4
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    fx = tf.placeholder(tf.float32, feat_vec_size)
    # Other initializers like tf.random_normal_initializer are possible too
    V1 = tf.get_variable("V1", [embedding_size1, feat_vec_size],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.
    z1 = tf.sigmoid(tf.tensordot(V1, fx, 1))
    V2 = tf.get_variable("V2", [embedding_size2, embedding_size1],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.
    z2 = tf.sigmoid(tf.tensordot(V2, z1, 1))
    W = tf.get_variable("W", [num_classes, embedding_size2])
    probs = tf.nn.softmax(tf.tensordot(W, z2, 1))
    # This is the actual prediction -- not used for training but used for inference
    one_best = tf.argmax(probs)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, 1)
    # Convert a value-based representation (e.g., [2]) into the one-hot representation ([0, 0, 1])
    # Because label is a tensor of dimension one, the one-hot is actually [[0, 0, 1]], so
    # we need to flatten it.
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))

    # TRAINING ALGORITHM CUSTOMIZATION
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    decay_steps = 8000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 100
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
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z2],
                                                                                  feed_dict={fx: np.mean(np.array([word_vectors.vectors[int(wordindex)] for wordindex in train_mat[ex_idx]]),0)})
            if (train_labels_arr[ex_idx] == pred_this_instance):
                train_correct += 1
            #print "Example " + repr(train_mat[ex_idx]) + "; gold = " + repr(train_labels_arr[ex_idx]) + "; pred = " + \
                  #repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            #print "  Hidden layer activations for this example: " + repr(z_this_instance)
        print repr(train_correct) + "/" + repr(len(train_labels_arr)) + " correct after training"

        # Evaluate on the dev set
        valid_correct = 0
        for ex_idx in xrange(0, len(valid_mat)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z2],
                                                                                  feed_dict={fx: np.mean(np.array([word_vectors.vectors[int(wordindex)] for wordindex in valid_mat[ex_idx]]),0)})
            if (valid_labels_arr[ex_idx] == pred_this_instance):
                valid_correct += 1
        print repr(valid_correct) + "/" + repr(len(valid_labels_arr)) + " correct for dev"

        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_mat)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z2],
                                                                                  feed_dict={fx: np.mean(np.array([word_vectors.vectors[int(wordindex)] for wordindex in test_mat[ex_idx]]),0)})
            test_results.append(SentimentExample(test_exs[ex_idx].indexed_words, pred_this_instance))
            if (test_labels_arr[ex_idx] == pred_this_instance):
                test_correct += 1
        print repr(test_correct) + "/" + repr(len(test_labels_arr)) + " correct for test"

    return test_results


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy2(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    word_vector_dimension = 300
    test_results = []

    train_mat, train_seq_lens, train_labels_arr = processdata(train_exs, seq_max_len)
    valid_mat, valid_seq_lens, valid_labels_arr = processdata(dev_exs, seq_max_len)
    test_mat, test_seq_lens, test_labels_arr = processdata(test_exs, seq_max_len)

    # MAKE THE DATA
    # Define some constants
    feat_vec_size = seq_max_len
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2
    lstmUnits = 64

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    fx = tf.placeholder(tf.int32, feat_vec_size)

    train_data = tf.Variable(tf.zeros([1, seq_max_len, word_vector_dimension]), dtype=tf.float32)
    train_data = tf.reshape(tf.nn.embedding_lookup(word_vectors.vectors, fx),[1, seq_max_len, word_vector_dimension])

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, train_data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    # This is the actual prediction -- not used for training but used for inference
    one_best = tf.argmax(prediction)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, 1)
    # Convert a value-based representation (e.g., [2]) into the one-hot representation ([0, 0, 1])
    # Because label is a tensor of dimension one, the one-hot is actually [[0, 0, 1]], so
    # we need to flatten it.
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(prediction, label_onehot, 1)))

    # TRAINING ALGORITHM CUSTOMIZATION
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    decay_steps = 8000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 100
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
                    fx: train_mat[ex_idx].astype(int),
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
            [probs_this_instance, pred_this_instance] = sess.run([prediction, one_best],
                                                                                  feed_dict={fx: train_mat[ex_idx].astype(int)})
            if (train_labels_arr[ex_idx] == pred_this_instance):
                train_correct += 1
            #print "Example " + repr(train_mat[ex_idx]) + "; gold = " + repr(train_labels_arr[ex_idx]) + "; pred = " + \
                  #repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            #print "  Hidden layer activations for this example: " + repr(z_this_instance)
        print repr(train_correct) + "/" + repr(len(train_labels_arr)) + " correct after training"

        # Evaluate on the dev set
        valid_correct = 0
        for ex_idx in xrange(0, len(valid_mat)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([prediction, one_best],
                                                                                  feed_dict={fx:valid_mat[ex_idx].astype(int)})
            if (valid_labels_arr[ex_idx] == pred_this_instance):
                valid_correct += 1
        print repr(valid_correct) + "/" + repr(len(valid_labels_arr)) + " correct for dev"

        # Evaluate on the test set
        test_correct = 0
        for ex_idx in xrange(0, len(test_mat)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance] = sess.run([prediction, one_best],
                                                                                  feed_dict={test_mat[ex_idx].astype(int)})
            test_results.append(SentimentExample(test_exs[ex_idx].indexed_words, pred_this_instance))
            if (test_labels_arr[ex_idx] == pred_this_instance):
                test_correct += 1
        print repr(test_correct) + "/" + repr(len(test_labels_arr)) + " correct for test"

    return test_results

from random import randint
def getTrainBatch(train_mat, train_labels_arr, batchSize, maxSeqLength):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(0, len(train_mat)-1)
        arr[i]=train_mat[num]
        labels.append(train_labels_arr[num])
    return arr, labels

def getTestBatch(test_mat, test_labels_arr, batchSize, maxSeqLength):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(0, len(test_mat) - 1)
        arr[i] = test_mat[num]
        labels.append(test_labels_arr[num])
    return arr, labels


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    maxSeqLength = 60
    numDimensions = 300
    batchSize = 24
    lstmUnits = 64
    numClasses = 2
    iterations = 1000
    test_results = []

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    train_mat, train_seq_lens, train_labels_arr = processdata(train_exs, maxSeqLength)
    valid_mat, valid_seq_lens, valid_labels_arr = processdata(dev_exs, maxSeqLength)
    test_mat, test_seq_lens, test_labels_arr = processdata(test_exs, maxSeqLength)

    train_data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    train_data = tf.nn.embedding_lookup(word_vectors.vectors, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, train_data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch(train_mat, train_labels_arr, batchSize, maxSeqLength)
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if (i % (iterations/10) == 0):
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if (i % (iterations/10) == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    iterations = 10
    for i in range(iterations):
        nextBatch, nextBatchLabels = getTestBatch(test_mat, test_labels_arr, batchSize, maxSeqLength)
        print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)