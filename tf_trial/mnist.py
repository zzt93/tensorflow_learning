from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def output_csv(prediction):
    import csv
    with open("output.csv", 'wb') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerow(['id', 'label'])
        for i in range(len(prediction)):
            wr.writerow([i, prediction[i]])


def mnist():
    """
    construct a simple two layer NN:
    - input layer with 784 neurons
    - output layer with 10 neurons, using softmax as activation function
    and cross entropy for error function
    :return:
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # None means that a dimension can be of any length.
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    w = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32)
    b = tf.Variable(tf.zeros([10]), dtype=tf.float32)
    # z = w * x + b
    z = tf.matmul(x, w) + b
    y_hat = tf.nn.softmax(z)

    # cost function
    y = tf.placeholder(tf.float32, [None, 10])
    # it is numerically unstable
    # -tf.reduce_sum(tf.log(y_hat) * y, reduction_indices=[1])
    # not use mean will reduce the accuracy to 0.5
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

    # construct the computation graph for evaluation
    # correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    prediction = sess.run(tf.argmax(y_hat, 1), feed_dict={x: mnist.test.images})
    output_csv(prediction)
    # print(prediction)
    # print(sess.run(correct_prediction, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


if __name__ == '__main__':
    mnist()
