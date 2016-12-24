## This module implements a sparse fit mechanism using Tensorflow.
import tensorflow as tf

import numpy as np
from scipy.stats import zscore
import time

def get_training_batches(X_tr, Y_tr, batch_size=200, delay=0, verbose=True):
    """
    Given an object of training data return a generator that gives training
    data batches.
    """
    idx = 0
    num_pts = np.shape(X_tr)[0] - delay
    Y_tr = np.concatenate((Y_tr[delay:, :], Y_tr[:delay, :]), axis=0)
    if verbose:
        print "Shape of training set: " + str(np.shape(X_tr))
        print "Shape of test set: " + str(np.shape(Y_tr))
    assert np.shape(X_tr)[0] == np.shape(Y_tr)[0]
    i = 0
    while True:
        if i + batch_size >= num_pts:
            remaining_pts = num_pts - i

            num_remove = batch_size - remaining_pts
            X_batch = np.concatenate((X_tr[-remaining_pts:, :],
                                      X_tr[:num_remove, :]), axis=0)
            Y_batch = np.concatenate((Y_tr[-remaining_pts:, :],
                                      Y_tr[:num_remove, :]), axis=0)
            yield (X_batch, Y_batch)
            i = num_remove
        else:
            yield (X_tr[i:i+batch_size, :],
                   Y_tr[i:i+batch_size, :])
            i += batch_size

def pearson_correllations(y, y_pred):
    """Computes the pearson correllation of a set of predictions.

    Parameters
    ----------
    y:
        Actual test set values. A numpy array of shape (npoints, nvariables)
    y_pred:
        The predicted values. A numpy array of shape (npoints, nvariables)
    """
    return (zscore(y)*zscore(y_pred)).mean(axis=0)

def lsq_loss(y_predicted, y_actual, name='test_loss'):
    """Simple least-squares loss without regularization. Useful for evaluating
    performance."""
    with tf.name_scope(name):
        loss = tf.reduce_sum(tf.square(tf.sub(y_actual, y_predicted)))
    return loss

def l1_loss(lambda_l1, weights, name='l1_loss'):
    """Least squares loss along with L1 regularization."""
    with tf.name_scope(name):
        l1_penalty = tf.mul(lambda_l1, tf.reduce_mean(tf.abs(weights)))
    return l1_penalty

def l1_group_loss(lambda_vec, weights, name='l1_loss'):
    """Least squares loss along with L1 regularization."""
    with tf.name_scope(name):
        l1_penalty = tf.reduce_mean(
                tf.matmul(tf.diag(lambda_vec),
                          tf.abs(weights)))
    return l1_penalty

def group_lasso_loss(lambda_vec, y_predicted, y_actual, weights):
    """Least squares with group LASSO"""
    with tf.name_scope("group_lasso_loss"):
        return tf.add(lsq_loss(y_predicted, y_actual),
                      l1_group_loss(lambda_vec, weights))

def threshold_by_percent_max(t, threshold, use_active_set=False):
    """Creates tensorflow ops to perform a thresholding of a tensor by a
    percentage of the maximum value. To be used when thresholding gradients.
    Optionally maintains an active set.

    Parameters
    ----------
    t: tensor
        The tensor to threshold by percent max.
    threshold: float
        A number between 0 and 1 that specifies the threshold.
    use_active_set: bool
        Specifies whether or not to use an active set.

    Returns
    -------
    A tensor of the same shape as t that has had all values under the threshold
    set to 0.
    """
    with tf.name_scope("threshold_by_percent_max"):
        # t = tf.convert_to_tensor(t, name="t")
        # shape = tf.shape(t)
        abs_t  = tf.abs(t)
        thresh_percentile = tf.constant(threshold, dtype=tf.float32)
        zeros = tf.zeros(shape=tf.shape(t), dtype=tf.float32)

        maximum = tf.reduce_max(abs_t, reduction_indices=[0])
        # A tensor, the same shape as t, that has the threshold values to be
        # compared against every value.
        thresh_one_voxel = tf.expand_dims(tf.mul(thresh_percentile,
                                                 maximum), 0)


        thresh_tensor = tf.tile(thresh_one_voxel,
                                tf.pack([tf.shape(t)[0], 1]))
        above_thresh_values = tf.greater_equal(abs_t, thresh_tensor)
        if use_active_set:
            active_set = tf.Variable(tf.equal(tf.ones(tf.shape(t)),
                                               tf.zeros(tf.shape(t))),
                                     name="active_set", dtype=tf.bool)

            active_set_inc = tf.assign(active_set,
                                       tf.logical_or(active_set,
                                                     above_thresh_values),
                                   name="incremented_active_set")
            active_set_size = tf.reduce_sum(tf.cast(active_set, tf.float32),
                                            name="size_of_active_set")
            return (tf.select(active_set_inc, t, zeros), active_set_size)
        else:
            return tf.select(above_thresh_values, t, zeros)

def group_lasso_fit(X_tr, Y_tr, X_test, Y_test, batch_size=100,
                     train_dir='.', max_iterations=350, learning_rate=0.0001,
                     l1_params=None, verbose=True):
        """Run group a LASSO fit using stochastic gradient descent on GPU.

        Parameters
        ----------
        X_tr:
            The training set. A numpy array of shape (time_points, #features).
        Y_tr:
            The training labels. A numpy array of shape (time_points, #labels)
            (The labels might be BOLD signal across different voxels,
            for example.)
        X_test:
            The test set. A numpy array of shape (time_points, #features). The
            number of time points may be different between the training and
            test sets, but the number of features may not.
        Y_test:
            The test labels. A numpy array of shape (time_points, #labels)
        batch_size: int, 100
            The minibatch size for the stochastic gradient descent.
        train_dir: str, '.'
            The directory that model and log directories will be saved in.
            This directory is important if you want to view the fit in
            Tensorboard.
        max_iterations: int, 350
            Early stopping is not yet implemented. Right now, you simply set
            a number of iterations.
        learning_rate: float, 0.0001
            The SGD learning rate.
        l1_params:
            A numpy array with length equal to the number of features, that
            specifies L1 regularization constants for every feature.

        Returns
        -------
        A dictionary containing both 'weights' and 'predictions', which are
        numpy arrays containing the learned weights, and the pearson
        correllation for each label, respectively.
        """
        if len(np.shape(Y_tr)) <= 1:
            num_predictors = 1
        else:
            num_predictors = np.shape(Y_tr)[1]
        num_features = np.shape(X_test)[1]

        if verbose:
            print "Building computation graph..."
        with tf.Graph().as_default():
            with tf.name_scope('test_data'):
                x_tst = tf.placeholder(tf.float32, shape=np.shape(X_test))
                y_tst = tf.placeholder(tf.float32, shape=np.shape(Y_test))
            with tf.name_scope('input_data'):
                x_tr = tf.placeholder(tf.float32,
                                      shape=(batch_size, num_features))
                y_tr = tf.placeholder(tf.float32,
                                      shape=(batch_size, num_predictors))

            # Getting the Y predictions
            W = tf.Variable(tf.zeros([num_features, num_predictors]),
                            name='weights')
            y_pred = tf.matmul(x_tr, W)


            # Loss function associated summaries.
            if l1_params is None:
                lambda_vec = tf.constant(tf.zeros([num_features],
                                                  dtype=tf.float32))
            else:
                assert len(l1_params) == num_features
                lambda_vec = tf.constant(l1_params, dtype=tf.float32)
            loss = group_lasso_loss(lambda_vec, y_pred, y_tr, W)
            loss_summary = tf.scalar_summary("Test LASSO Loss", loss)

            # Weight histogram summary.
            W_hist_summary = tf.histogram_summary("Weight Distribution", W)

            # Optimizer
            global_step = tf.Variable(0, name='global_step', dtype=tf.int64)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            op = optimizer.minimize(loss)

            train_summary = tf.merge_summary([W_hist_summary, loss_summary])

            # Report the test set error (but don't optimize on it)
            y_pred_test = tf.matmul(x_tst, W)
            test_loss = lsq_loss(y_pred_test, y_tst, name='test_loss')
            tst_loss_summary = tf.scalar_summary("Test LSQ Loss", test_loss)

            # Tensorflow boilerplate
            init = tf.initialize_all_variables()

            saver = tf.train.Saver({'weights': W})
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            summary_writer = tf.train.SummaryWriter(train_dir + '/log',
                                                    sess.graph)

            if verbose:
                print "initializing variables..."
            sess.run(init)
            batch_gen = get_training_batches(X_tr, Y_tr, batch_size=batch_size,
                                             verbose=verbose)

            # Run training.
            start_time = time.time()
            for step in xrange(max_iterations):
                if verbose:
                    print('\n== Optimization Step %d of %d =='
                          % (step, max_iterations))
                training_loss = 0.
                if verbose:
                    print "...Generating batch"
                X_batch, Y_batch = batch_gen.next()
                if verbose:
                    print "...Calculating Gradients"
                (_, loss_eval, train_summ_eval) = sess.run(
                        [op, loss, train_summary], feed_dict={x_tr: X_batch,
                                                              y_tr: Y_batch})

                # Write out a bunch of summaries:
                duration = time.time() - start_time
                if verbose:
                    print('Step %d: loss=%.2f (%.3f sec)'
                          % (step, loss_eval, duration))
                summary_writer.add_summary(train_summ_eval, step)
                summary_writer.flush()

                # We evaluate the test set every so often, just to visualize it
                # in tensorboard.
                if step > 0 and step % 10 == 0:
                    if verbose:
                        print "Shape of x_test: " + str(np.shape(X_test))
                        print "shape of y_test: " + str(np.shape(Y_test))

                    tst_summ, tst_loss = sess.run([tst_loss_summary, test_loss],
                            feed_dict={x_tst: X_test,
                                       y_tst: Y_test})
                    if verbose:
                        print "test_loss: " + str(tst_loss)
                        print('Step %d: loss=%.2f test_loss=%.2f (%.3f sec)'
                              % (step, training_loss, tst_loss, duration))
                    summary_writer.add_summary(tst_summ, step)
                    summary_writer.flush()

        fit = {}
        y_pred_test_ev = sess.run([y_pred_test], feed_dict={x_tst: X_test,
                                                            y_tst: Y_test})[0]

        final_W = sess.run([W])
        fit['weights'] = final_W
        # Evaluate Prediction Accuracy
        fit['predictions'] = pearson_correllations(Y_test, y_pred_test_ev)
        return fit

def threshold_gradient_descent_fit(X_tr, Y_tr, X_test, Y_test, batch_size=100,
                     train_dir='.', max_iterations=350, learning_rate=0.0001,
                     threshold_grad_desc = 0.5, use_active_set=True,
                     verbose=True):
        """Run Threshold gradient descent on GPU.

        Parameters
        ----------
        X_tr:
            The training set. A numpy array of shape (time_points, #features).
        Y_tr:
            The training labels. A numpy array of shape (time_points, #labels)
            (The labels might be BOLD signal across different voxels,
            for example.)
        X_test:
            The test set. A numpy array of shape (time_points, #features). The
            number of time points may be different between the training and test
            sets, but the number of features may not.
        Y_test:
            The test labels. A numpy array of shape (time_points, #labels)
        batch_size: int, 100
            The minibatch size for the stochastic gradient descent.
        train_dir: str, '.'
            The directory that model and log directories will be saved in.
            This directory is important if you want to view the fit in
            Tensorboard.
        max_iterations: int, 350
            Early stopping is not yet implemented. Right now, you simply set
            a number of iterations.
        learning_rate: float, 0.0001
            The SGD learning rate.
        threshold_grad_desc: float, 0.5
            A float between 0 and 1 specifying how to threshold the gradients.
            A value of 0.dd will only allow gradients of magnitude that are
            dd% of the maximum gradient's absolute value.
        use_active_set: bool, True
            Specifies whether or not to use an active set while fitting.
        verbose: bool, True
            Specifies whether or not to use

        Returns
        -------
        A dictionary containing both 'weights' and 'predictions', which are
        numpy arrays containing the learned weights, and the pearson
        correllation for each label, respectively.
        """
        if len(np.shape(Y_tr)) <= 1:
            num_predictors = 1
        else:
            num_predictors = np.shape(Y_tr)[1]
        num_features = np.shape(X_test)[1]

        if verbose:
            print "Building computation graph..."
        with tf.Graph().as_default():
            with tf.name_scope('test_data'):
                x_tst = tf.placeholder(tf.float32, shape=np.shape(X_test))
                y_tst = tf.placeholder(tf.float32, shape=np.shape(Y_test))
            with tf.name_scope('input_data'):
                x_tr = tf.placeholder(tf.float32,
                                      shape=(batch_size, num_features))
                y_tr = tf.placeholder(tf.float32,
                                      shape=(batch_size, num_predictors))

            # Getting the Y predictions
            W = tf.Variable(tf.zeros([num_features, num_predictors]),
                            name='weights')
            y_pred = tf.matmul(x_tr, W)

            # Loss function associated summaries.
            loss = lsq_loss(y_pred, y_tr, name='training_lsq_loss')
            loss_summary = tf.scalar_summary("Training Loss", loss)

            # Weight histogram summary.
            W_hist_summary = tf.histogram_summary("Weight Distribution", W)

            # Optimizer
            global_step = tf.Variable(0, name='global_step', dtype=tf.int64)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            W_grad, W_var = optimizer.compute_gradients(loss, [W])[0]

            # Normalize the gradient.
            norm_grads = tf.nn.l2_normalize(W_grad, dim=0,
                                              name="normalize_gradients")
            # Apply threshold gradient descent.
            thresholded_grads, active_set_count = threshold_by_percent_max(
                        norm_grads, threshold_grad_desc, use_active_set=True)
            thresh_grad_hist_summary = tf.histogram_summary(
                                  "Thresholded Gradients",
                                  thresholded_grads)
            # Keep track of the active set used for TGD.
            active_set_count_summary = tf.scalar_summary(
                                        "Number of Active Gradients",
                                        active_set_count)
            # Run this op to apply all the accumulated gradients.
            op = optimizer.apply_gradients([(thresholded_grads, W_var)],
                                            global_step=global_step,
                                            name="apply_accumulated_grads")
            train_summary = tf.merge_summary([thresh_grad_hist_summary,
                                              active_set_count_summary,
                                              W_hist_summary, loss_summary])

            # Report the test set error (but don't optimize on it)
            y_pred_test = tf.matmul(x_tst, W)
            test_loss = lsq_loss(y_pred_test, y_tst, name='test_loss')
            tst_loss_summary = tf.scalar_summary("Test LSQ Loss", test_loss)

            # Tensorflow boilerplate
            init = tf.initialize_all_variables()

            saver = tf.train.Saver({'weights': W})
            sess = tf.Session(config=tf.ConfigProto(
                                  log_device_placement=verbose))
            summary_writer = tf.train.SummaryWriter(train_dir + '/log',
                                                    sess.graph)

            if verbose:
                print "initializing variables..."
            sess.run(init)
            batch_gen = get_training_batches(X_tr, Y_tr, batch_size=batch_size,
                                             verbose=verbose)

            # Run training.
            start_time = time.time()
            for step in xrange(max_iterations):
                if verbose:
                    print('\n== Optimization Step %d of %d =='
                          % (step, max_iterations))
                training_loss = 0.
                if verbose:
                    print "...Generating batch"
                X_batch, Y_batch = batch_gen.next()
                if verbose:
                    print "...Calculating Gradients"
                (_, loss_eval, train_summ_eval) = sess.run(
                        [op, loss, train_summary], feed_dict={x_tr: X_batch,
                                                              y_tr: Y_batch})

                # Write out a bunch of summaries:
                duration = time.time() - start_time
                if verbose:
                    print('Step %d: loss=%.2f (%.3f sec)'
                          % (step, loss_eval, duration))
                summary_writer.add_summary(train_summ_eval, step)
                summary_writer.flush()

                # We evaluate the test set every so often, just to visualize it
                # in tensorboard.
                if step > 0 and step % 10 == 0:
                    if verbose:
                        print "Shape of x_test: " + str(np.shape(X_test))
                        print "shape of y_test: " + str(np.shape(Y_test))

                    tst_summ, tst_loss = sess.run([tst_loss_summary, test_loss],
                            feed_dict={x_tst: X_test,
                                       y_tst: Y_test})
                    if verbose:
                        print "test_loss: " + str(tst_loss)
                        print('Step %d: loss=%.2f test_loss=%.2f (%.3f sec)'
                              % (step, training_loss, tst_loss, duration))
                    summary_writer.add_summary(tst_summ, step)
                    summary_writer.flush()

        fit = {}
        y_pred_test_ev = sess.run([y_pred_test], feed_dict={x_tst: X_test,
                                                            y_tst: Y_test})[0]

        final_W = sess.run([W])
        fit['weights'] = final_W
        # Evaluate Prediction Accuracy
        fit['predictions'] = pearson_correllations(Y_test, y_pred_test_ev)
        return fit
