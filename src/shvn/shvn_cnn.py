import matplotlib.pylab as plt
import numpy as np
import scipy.io
import sklearn
import tensorflow as tf
from skimage import color

"""
http://ufldl.stanford.edu/housenumbers/

https://www.tensorflow.org/tutorials/estimators/cnn
"""


class SHVN:
    NUM_CLASSES = 10  # 0..9

    def read_data(self, path):
        print('Analyze ' + str(path))
        mat = scipy.io.loadmat(path)
        X = mat['X']
        print('X shape: ' + str(X.shape))

        y = mat['y']
        print('y shape: ' + str(y.shape))

        return X, y

    def rgb2bw(self, X):
        """
        Convert 3d image to 2d image
        :param X:
        :return:
        """
        W, H, C, N = X.shape
        Xbw = np.zeros((N, W, H))

        for i in range(N):
            Xbw[i] = np.array(color.rgb2gray(X[:, :, :, i]))

        return Xbw

    def category2indicator(self, y):
        """
        10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
        :param y:
        :return:
        """
        Y = np.zeros(shape=(y.shape[0], self.NUM_CLASSES))

        for idx, item in enumerate(y):
            if item == 10:
                Y[idx][0] = 1
            else:
                Y[idx][item] = 1

        return Y

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epoch=50, learning_rate=0.001, batchsz=100):

        tf_X = tf.placeholder(dtype=tf.float32)
        tf_Y = tf.placeholder(dtype=tf.float32)

        # Convolutional layer
        num_channels = 32
        W, H = Xtrain[0].shape

        input_layer = tf.reshape(tf_X, [-1, W, H, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,  # [batch_size, image_height, image_width, channels] = [batch_size, W, H, 1]
            filters=num_channels,
            kernel_size=[5, 5],
            padding="same")  # Output: [batch_size, W, H, num_channels]

        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)  # Output: [batch_size, W // 2, H // 2, num_channels]

        # Dense layers
        D = W // 2 * H // 2 * num_channels
        pool1_flat = tf.reshape(pool1, [-1, D])

        # Use two hidden units
        M1 = 1000
        M2 = 500

        tf_W1 = tf.Variable(dtype=tf.float32,
                            initial_value=tf.random.normal(shape=(D, M1), mean=0, stddev=tf.math.sqrt(1 / D)))
        tf_b1 = tf.Variable(dtype=tf.float32, initial_value=np.zeros(shape=(M1)))

        tf_W2 = tf.Variable(dtype=tf.float32,
                            initial_value=tf.random.normal(shape=(M1, M2), mean=0, stddev=tf.math.sqrt(1 / M1)))
        tf_b2 = tf.Variable(dtype=tf.float32, initial_value=np.zeros(shape=(M2)))

        tf_W3 = tf.Variable(dtype=tf.float32,
                            initial_value=tf.random.normal(shape=(M2, self.NUM_CLASSES), mean=0,
                                                           stddev=tf.math.sqrt(1 / M2)))
        tf_b3 = tf.Variable(dtype=tf.float32, initial_value=np.zeros(shape=(self.NUM_CLASSES)))

        tf_Z1 = tf.nn.relu(tf.matmul(pool1_flat, tf_W1) + tf_b1)

        tf_Z2 = tf.nn.relu(tf.matmul(tf_Z1, tf_W2) + tf_b2)
        tf_Yhat = tf.nn.softmax(tf.matmul(tf_Z2, tf_W3) + tf_b3)

        tf_cost = tf.reduce_sum(-1 * tf_Y * tf.math.log(tf_Yhat + 1e-5))
        tf_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
            tf_cost)

        training_accuracies = []
        test_accuracies = []
        epoches = []
        N = Xtrain.shape[0]

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            nBatches = np.math.ceil(N / batchsz)

            for i in range(epoch):
                epoches.append(i)
                Xtrain, Ytrain = sklearn.utils.shuffle(Xtrain, Ytrain)

                for j in range(nBatches):
                    lower = j * batchsz
                    upper = np.min([(j + 1) * batchsz, N])

                    session.run(tf_train,
                                feed_dict={tf_X: Xtrain[lower:upper], tf_Y: Ytrain[lower: upper]})

                test_error, Yhat = session.run([tf_cost, tf_Yhat], feed_dict={tf_X: Xtest, tf_Y: Ytest})
                test_accuracy = self.score(Ytest, Yhat)
                test_accuracies.append(test_accuracy)

                train_error, Yhat = session.run([tf_cost, tf_Yhat], feed_dict={tf_X: Xtrain, tf_Y: Ytrain})
                training_accuracy = self.score(Ytrain, Yhat)
                training_accuracies.append(training_accuracy)

                print('Epoch ' + str(i) + ' / test error = ' + str(test_error / Xtest.shape[0])
                      + ' / train error = ' + str(train_error / Xtrain.shape[0])
                      + ' / training_accuracy = ' + str(training_accuracy)
                      + ' / test_accuracy = ' + str(test_accuracy))

        # plot
        print(training_accuracies)
        print(test_accuracies)

        plt.plot(epoches, training_accuracies)
        plt.plot(epoches, test_accuracies)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.title('Accuracy')
        plt.legend()
        plt.show()

    def score(self, Y, Yhat):
        y = np.argmax(Y, axis=1)
        yhat = np.argmax(Yhat, axis=1)
        return np.mean(y == yhat)


def main():
    shvn = SHVN()

    COLAB_PATH = '/content/drive/My Drive/Colab Notebooks/'
    LOCAL_PATH = '../../data/SVHN/'
    USING_PATH = LOCAL_PATH

    if USING_PATH == COLAB_PATH:
        from google.colab import drive
        drive.mount('/content/drive')

    Xtrain, ytrain = shvn.read_data(USING_PATH + 'test_32x32.mat')
    Xtrain_bw = shvn.rgb2bw(Xtrain).astype('float32')
    print('Xtrain_bw: ' + str(Xtrain_bw.shape))
    Ytrain = shvn.category2indicator(ytrain).astype('float32')
    print('Ytrain: ' + str(Ytrain.shape))

    Xtest, ytest = shvn.read_data(USING_PATH + 'test_32x32.mat')
    Xtest_bw = shvn.rgb2bw(Xtest).astype('float32')
    Ytest = shvn.category2indicator(ytest).astype('float32')

    limit = None
    shvn.fit(Xtrain_bw[:limit], Ytrain[:limit], Xtest_bw, Ytest)


main()
