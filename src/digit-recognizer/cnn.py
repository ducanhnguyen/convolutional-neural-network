import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

"""
test accuracy = 0.98728

Data: https://www.kaggle.com/c/digit-recognizer/leaderboard#score
"""


class Digit_Recognizer:
    NUM_CLASSES = 10  # 0..9

    def fit(self, Xtrain, Ytrain, Xtest, epoch=30, learning_rate=0.001, batchsz=100):
        tf_X = tf.placeholder(dtype=tf.float32)
        tf_Y = tf.placeholder(dtype=tf.float32)

        # Convolutional layer
        W, H, _ = Xtrain[0].shape
        input_layer = tf.reshape(tf_X, [-1, W, H, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,  # [batch_size, 32, 32, 1]
            filters=6,
            kernel_size=[5, 5],
            padding="valid",
            activation='relu')  # Output: [batch_size, 28, 28, 6]

        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)  # Output: [batch_size, 14, 14, 6]

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=16,
            kernel_size=[5, 5],
            padding="valid",
            activation='relu')  # Output: [batch_size, 10, 10, 16]

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)  # Output: [batch_size, 5, 5, 16]

        D = 5 * 5 * 16
        pool2_flat = tf.reshape(pool2, [-1, D])

        # Dense layers
        M1 = 120
        M2 = 84
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

        tf_Z1 = tf.nn.relu(tf.matmul(pool2_flat, tf_W1) + tf_b1)

        tf_Z2 = tf.nn.relu(tf.matmul(tf_Z1, tf_W2) + tf_b2)

        tf_Yhat = tf.nn.softmax(tf.matmul(tf_Z2, tf_W3) + tf_b3)

        tf_cost = tf.reduce_sum(-1 * tf_Y * tf.math.log(tf_Yhat + 1e-5))

        tf_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
            tf_cost)

        training_accuracies = []
        epoches = []
        N = Xtrain.shape[0]

        session = tf.Session()
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

            train_error, Yhat = session.run([tf_cost, tf_Yhat], feed_dict={tf_X: Xtrain, tf_Y: Ytrain})
            training_accuracy = self.score(Ytrain, Yhat)
            training_accuracies.append(training_accuracy)
            print('Epoch ' + str(i)
                  + ' / train error = ' + str(train_error / Xtrain.shape[0])
                  + ' / training_accuracy = ' + str(training_accuracy))

        # plot
        print(training_accuracies)

        plt.plot(epoches, training_accuracies)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.title('Accuracy')
        plt.legend()
        plt.show()

        # Predict and export to csv
        import csv
        yhat = session.run(tf_Yhat, feed_dict={tf_X: Xtest})
        yhat = np.argmax(yhat, axis=1)
        print(yhat)
        with open('/content/drive/My Drive/Colab Notebooks/digit-recognizer/submission.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['ImageId', 'Label'])

            for idx, row in enumerate(yhat):
                writer.writerow([idx + 1, row])

        csvFile.close()

    def score(self, Y, Yhat):
        y = np.argmax(Y, axis=1)
        yhat = np.argmax(Yhat, axis=1)
        return np.mean(y == yhat)

    def convert2indicator(self, y):
        '''
        Convert category into indicator matrix
        :param y:
        :return:
        '''
        max_value = np.amax(y) + 1
        Y = np.zeros((len(y), max_value))
        for idx, value in enumerate(y):
            Y[idx, value] = 1
        return Y

    def readTrainingDigitRecognizer(self, pathStr, limit=None):
        data = pd.read_csv(pathStr)
        data = data.to_numpy()  # convert to ndarray, headers removed
        X = data[:limit, 1:] * 1.0 / 255  # convert to the range of [0..1] (normalization step)
        y = data[:limit, 0]  # labels
        print('Load data from ' + str(pathStr) + ' done')
        return X, y

    def readTestingDigitRecognizer(self, pathStr, limit=None):
        data = pd.read_csv(pathStr)
        data = data.to_numpy()  # convert to ndarray, headers removed
        X = data[:limit] * 1.0 / 255  # convert to the range of [0..1] (normalization step)
        print('Load data from ' + str(pathStr) + ' done')
        return X


def main():
    dr = Digit_Recognizer()

    COLAB_PATH = '/content/drive/My Drive/Colab Notebooks/digit-recognizer/'
    LOCAL_PATH = '../../data/digit-recognizer/'
    USING_PATH = COLAB_PATH

    if USING_PATH == COLAB_PATH:
        from google.colab import drive
        drive.mount('/content/drive/')

    Xtrain, ytrain = dr.readTrainingDigitRecognizer(USING_PATH + 'train.csv')
    Xtrain = Xtrain.astype('float32')
    print('Xtrain: ' + str(Xtrain.shape))
    Xtrain = Xtrain.reshape((-1, 28, 28, 1))
    print('Reshaped. Xtrain: ' + str(Xtrain.shape))

    # Make it to 32x32
    Xtrain = np.pad(Xtrain, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    print('Padded. Xtrain: ' + str(Xtrain.shape))

    Ytrain = dr.convert2indicator(ytrain).astype('float32')
    print('Ytrain: ' + str(Ytrain.shape))

    Xtest = dr.readTestingDigitRecognizer(USING_PATH + 'test.csv')
    Xtest = Xtest.astype('float32')
    print('Xtest: ' + str(Xtest.shape))
    Xtest = Xtest.reshape((-1, 28, 28, 1))
    print('Reshaped. Xtest: ' + str(Xtest.shape))
    Xtest = np.pad(Xtest, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    print('Padded. Xtest: ' + str(Xtest.shape))

    limit = None
    dr.fit(Xtrain[:limit], Ytrain[:limit], Xtest)


main()
