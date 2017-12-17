import tensorflow as tf
import numpy as np


def add_triple(sess):
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
    triple = adder_node * 3
    print(sess.run(triple, {a: 3, b: 4.5}))


def linear(sess):
    w = tf.Variable([-.5], dtype=tf.float32)
    b = tf.Variable([.5], dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32)
    init = tf.global_variables_initializer()
    sess.run(init)
    linear_model = w * x + b
    print sess.run(linear_model, {x: [1, 2, 3, 4]})

    y = tf.placeholder(tf.float32)
    square_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(square_deltas)
    # run with graph and input
    print sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess.run(init)
    for i in range(1000):
        # run with optimizer and input
        # same small data, multiple times
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print sess.run([w, b])


def estimator_linear(sess):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
    # TensorFlow provides many helper methods to read and set up data sets.
    # Here we use two data sets: one for training and one for evaluation
    # We have to tell the function how many batches
    # of data (num_epochs) we want and how big each batch should be.
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

    estimator.train(input_fn=input_fn, steps=1000)
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("train metrics: %r" % train_metrics)
    print("eval metrics: %r" % eval_metrics)


def custom_est():
    def model_fn(features, labels, mode):
        w = tf.get_variable("w", [1], dtype=tf.float32)
        b = tf.get_variable("b", [1], dtype=tf.float32)
        y = w * features['x'] + b

        loss = tf.reduce_sum(tf.square(y - labels))

        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(global_step, 1), tf.assign_add(global_step, 1))
        return tf.estimator.EstimatorSpec(mode, y, loss, train)

    estimator = tf.estimator.Estimator(model_fn=model_fn)
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7., 0.])
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

    # train
    estimator.train(input_fn=input_fn, steps=1000)
    # Here we evaluate how well our model did.
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("train metrics: %r" % train_metrics)
    print("eval metrics: %r" % eval_metrics)


if __name__ == '__main__':
    sess = tf.Session()
    # add_triple(sess)
    # linear(sess)
    estimator_linear(sess)
