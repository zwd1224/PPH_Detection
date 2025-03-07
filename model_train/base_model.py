import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_regression
class Model(object):
    """ General model class. """
    def __init__(self, directory, learning_rate, num_filters, num_features, frame_size, num_samples=1,strength=80):
        """ Initializes the model and creates a new seesion.

            Parameters
            ----------
            directory : str
                The folder where the model is stored.
            learning_rate : float
                Defines the learning rate for gradient descent.

        """
        self.directory = directory
        # Create a tensorflow session for the model
        self.sess = tf.Session()
        # Define the learning rate for the neural net
        self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
        #                                                 global_step=self.global_step,
        #                                                 decay_steps=1000,
        #                                                 decay_rate=0.96,
        #                                                 staircase=True)
        self.learning_rate = learning_rate
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout')

        self.num_filters = num_filters
        self.num_features = num_features
        self.num_classes = 2
        self.frame_size = frame_size
        self.num_samples = num_samples
        # Define placeholder for the inputs X, the labels y and the users feedback
        self.label = tf.placeholder(shape=[None, self.num_classes], name='y', dtype=tf.float32)
        self.X = tf.placeholder(shape=[None, self.num_features], name="X", dtype=tf.float32)
        self.attention = tf.placeholder(shape=[None, self.num_features], name="a", dtype=tf.float32)
        self.feedback = tf.placeholder(shape=[None, self.num_features], name="feedback", dtype=tf.float32)
        # Create a placeholder that defines the phase: training or testing
        self.phase = tf.placeholder( name='phase', dtype=tf.float32)
        self.saved_model = None
        self.saved_weight = None
        self.strength=strength
        self.path_length=10
    def explain(self, X):
        feed_dict = {}
        feed_dict['phase:0'] = 0
        feed_dict['X:0'] = X
        # Run the evaluation
        probs = self.sess.run(tf.reduce_mean(self.probs, axis=0),
                              feed_dict=feed_dict)
        return probs

    def pre_train(self, X, y, dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['phase:0'] = 2
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # The attention tensor is set to ones during pre-training
        feed_dict['a:0'] = np.ones([X.shape[0], self.num_features])
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        # Run the training step and write the results to Tensorboard

        summary, _ = self.sess.run([self.merged, self.pre_train_step],
                                   feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        # summary = self.sess.run(self.merged_images,
        #                         feed_dict={self.X: X[0:10], self.feedback: np.ones([10, self.num_features]), self.attention: np.ones([10, self.num_features]), self.phase: 2})
        # self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')

    def train(self, X, y, dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['phase:0'] = 1
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # 遍历字典并打印值的数据类型
        # for key, value in feed_dict.items():
        #     data_type = type(value)
        #     print(f"{key} Data Type:", data_type)
        # Sample the attention vector from the Bernoulli distribution
        attention = self.sess.run(tf.reshape(self.distribution.sample(self.num_samples), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        # Run the training step and write the results to Tensorboard

        summary, _ = self.sess.run([self.merged, self.train_step],
                                   feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        # summary = self.sess.run(self.merged_images,
        #                         feed_dict={self.X: X[0:10],
        #                                    self.feedback: np.ones([10, self.num_features]), self.attention: attention[0:10], self.phase: 1})
        # self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')
    # cnn_tree_train
    def cnn_tree_train(self, X, y,dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['phase:0'] = 1
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # 遍历字典并打印值的数据类型
        # for key, value in feed_dict.items():
        #     data_type = type(value)
        #     print(f"{key} Data Type:", data_type)
        # Sample the attention vector from the Bernoulli distribution
        attention = self.sess.run(tf.reshape(self.distribution.sample(self.num_samples), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        # Run the training step and write the results to Tensorboard

        summary, _ = self.sess.run([self.merged, self.cnn_tree_train_step],
                                   feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        # summary = self.sess.run(self.merged_images,
        #                         feed_dict={self.X: X[0:10],
        #                                    self.feedback: np.ones([10, self.num_features]), self.attention: attention[0:10], self.phase: 1})
        # self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')
    def feedback_tree_train(self, X, y, feedback, dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            feedback: numpy array [batch_size, num_features]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['phase:0'] = 1
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # Sample the attention vector from the Bernoulli distribution
        attention = self.sess.run(tf.reshape(self.distribution.sample(self.num_samples), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['feedback:0'] = feedback
        # Run the training step and write the results to Tensorboard
        # print('self.strength',self.strength)
        # print('strength',self.strength)

        summary, _ = self.sess.run([self.merged, self.feedback_tree_train_step],
                                   feed_dict=feed_dict)


        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        # summary = self.sess.run(self.merged_images,
        #                         feed_dict={self.X: X[0:10], self.feedback: feedback[0:10], self.attention: attention[0:10], self.phase: 1})
        # self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')
    def feedback_train(self, X, y, feedback, dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            feedback: numpy array [batch_size, num_features]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['phase:0'] = 1
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # Sample the attention vector from the Bernoulli distribution
        attention = self.sess.run(tf.reshape(self.distribution.sample(self.num_samples), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['feedback:0'] = feedback
        # Run the training step and write the results to Tensorboard

        summary, _ = self.sess.run([self.merged, self.feedback_train_step],
                                   feed_dict=feed_dict)

        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        # summary = self.sess.run(self.merged_images,
        #                         feed_dict={self.X: X[0:10], self.feedback: feedback[0:10], self.attention: attention[0:10], self.phase: 1})
        # self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')

    def evaluate(self, X, y, pre_trainining=False):
        """ Evaluates X and compares to the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            pre_trainining: boolean

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        if pre_trainining:
            phase = 2
            attention = np.ones([X.shape[0], self.num_features])
        else:
            phase = 0
            feed_dict['phase:0'] = phase
            attention = self.sess.run(tf.reshape(self.distribution.sample(), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['phase:0'] = phase
        feed_dict['a:0'] = attention
        # Run the evaluation and write the results to Tensorboard
        summary = self.sess.run(self.merged,feed_dict=feed_dict)
        self.test_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        # summary = self.sess.run(self.merged_images,
        #                         feed_dict={self.X: X[0:10], self.feedback: np.ones([10, self.num_features]), self.attention: attention[0:10], self.phase: phase})
        # self.test_writer.add_summary(summary, global_step=self.sess.run(self.global_step))

    def predict(self, X,flag):
        feed_dict = {}
        feed_dict["phase:0"] = 0
        feed_dict["X:0"] = X
        # Sample the attention vector from the Bernoulli distribution
        # flag 反馈计算准确度需要
        if flag==True:
            attention = self.sess.run(self.distribution.sample(), feed_dict=feed_dict)
            feed_dict['a:0'] = attention
            feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        # Run the evaluation
        return self.sess.run(self.predictions, feed_dict=feed_dict)

    def save(self):
        self.saver.save(self.sess, self.directory + '/model.ckpt', global_step=self.sess.run(self.global_step))

    def confusion_matrix(self, X, y,flag=False):
        predictions = self.predict(X,flag)
        threshold = 0.5
        # print('预测情况：')
        one_ = []
        zero_ = []
        # print('predictions')
        # print(predictions)
        for i in range(len(predictions)):
            if y[i][1] == 1:
                one_.append(predictions[i][1])
                # ones+=1
            else:
                zero_.append(predictions[i][1])
        # print(y_test)
        # print(sorted(one_))
        # print(sorted(zero_))
        print(np.mean(one_), np.median(one_),'阳性样本', len(one_))
        print(np.mean(zero_), np.median(zero_),'阴性样本', len(zero_))
        for i in range(predictions.shape[0]):
            if predictions[i, 1] > threshold:
                predictions[i, 0] = 0
                predictions[i, 1] = 1
            else:
                predictions[i, 0] = 1
                predictions[i, 1] = 0

        predictions = tf.argmax(predictions, axis=1)
        return self.sess.run(tf.confusion_matrix(labels=np.argmax(y, axis=1), predictions=predictions))

    def kappa(self, X, y):
        predictions = self.predict(X)
        predictions = np.argmax(predictions, axis=1)
        y = np.argmax(y, axis=1)
        return cohen_kappa_score(y, predictions)

    def add_logging(self, config=None):
        """ Creates summaries for accuracy, auc and loss. """

        # Calculate the accuracy for multiclass classification
        dynamic_num_samples = tf.floordiv(tf.shape(self.attention)[0], tf.shape(self.X)[0])
        label = tf.reshape(tf.tile(tf.reshape(self.label, [-1]), [dynamic_num_samples]), [-1, self.num_classes])
        correct_prediction = tf.equal(tf.argmax(label, axis=1), tf.argmax(self.predictions, axis=1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        # # Calculate the confusion matrix
        confusion = tf.confusion_matrix(labels=tf.argmax(label, axis=1), predictions=tf.argmax(self.predictions, axis=1))
        confusion = tf.cast(confusion, dtype=tf.float32)
        # Calculate the kappa accuracy
        sum0 = tf.reduce_sum(confusion, axis=0)
        sum1 = tf.reduce_sum(confusion, axis=1)
        expected_accuracy = tf.reduce_sum(tf.multiply(sum0, sum1)) / (tf.square(tf.reduce_sum(sum0)))
        expected_accuracy_summary = tf.summary.scalar('expected_accuracy', expected_accuracy)
        w_mat = np.zeros([self.num_classes, self.num_classes], dtype=np.int)
        w_mat += np.arange(self.num_classes)
        w_mat = np.abs(w_mat - w_mat.T)
        k = (self.accuracy - expected_accuracy) / (1 - expected_accuracy)
        # Calculate the precision
        _, precision = tf.metrics.precision(
                                    labels=label,
                                    predictions=self.predictions)
        # Add the summaries
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        # kappa_accuracy = tf.summary.scalar('kappa', k)
        # precision_summary = tf.summary.scalar('precision', precision)
        loss_summary = tf.summary.scalar('loss', self.loss)
        # learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)

        # Combine all the summaries
        self.merged = tf.summary.merge_all()
        # Create summary writers for train and test set
        self.train_writer = tf.summary.FileWriter(self.directory + '/train',
                                                  self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.directory + '/test')

    def build_net(self, input, layer_sizes, scope, phase, nonlinearity=tf.nn.relu, last_activation=tf.nn.softmax, logging=True, batch_norm=False, reuse=False,flag=False):
        """ Builds a dense DNN with the architecture defined by layer_sizes.

            Parameters
            ----------
            input: tf tensor
                The input to the neural net
            layer_sizes: list
                 List containing the size of all layers, including the input and output sizes.
                 [input_size, hidden_layers, output_size]
            scope: str
                The name of the variable scope, which refers to the type of model
            phase: tf placeholder (boolean)
                Holds the state of the model: training or testing
            nonlinearity: tf callable
                The nonlinearity applied between layers in the neural net
            last_activation: tf callable
                The nonlinearity that defines the output of the neural net
            logging: boolean
                Whether to log the weights in Tensorboard
            batch_norm: boolean
                Whether to apply batch normalization between layers
            reuse: boolean
                Whether the weights in neural net can be reused later on

            Returns
            ----------
            pre_activation: tf tensor
                The logits which are equivalent to the output of the DNN before the non-linearity
            predictions: tf tensor
                The output of the DNN
            reg_loss: tf tensor
                The regularization loss of the neural net

        """
        activation = input
        reg_loss = []
        # Define the DNN layers
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for i, layer_size in enumerate(layer_sizes[1::], 1):
                if i < len(layer_sizes) - 1:
                    W = tf.get_variable(
                            name='W{0}'.format(i),
                            shape=[layer_sizes[i-1], layer_size],
                            initializer=tf.contrib.layers.xavier_initializer())
                    reg_loss.append(tf.nn.l2_loss(W))
                    b = tf.Variable(
                            tf.zeros([layer_size]), name='b{0}'.format(i))
                    if logging:
                        summ = tf.summary.histogram('W{0}'.format(i), W)
                        summm = tf.summary.scalar('W{0}_sparsity'.format(i), tf.nn.zero_fraction(W))
                else:
                    with tf.variable_scope("last"):
                        W = tf.get_variable(
                                name='W{0}'.format(i),
                                shape=[layer_sizes[i-1], layer_size],
                                initializer=tf.contrib.layers.xavier_initializer())
                        reg_loss.append(tf.nn.l2_loss(W))
                        b = tf.Variable(
                                tf.zeros([layer_size]), name='b{0}'.format(i))
                        if logging:
                            summ = tf.summary.histogram('W{0}'.format(i), W)
                            summm = tf.summary.scalar('W{0}_sparsity'.format(i), tf.nn.zero_fraction(W))

                if i < len(layer_sizes) - 1:
                    pre_activation = tf.add(tf.matmul(activation, W), b,
                                            name='pre_activation{0}'.format(i))
                    activation = nonlinearity(pre_activation,
                                              name='activation{0}'.format(i))

                    activation = tf.nn.dropout(activation, keep_prob=self.keep_prob)
                    if batch_norm:
                        activation = tf.contrib.layers.batch_norm(activation,
                                                                  center=True,
                                                                  scale=True,
                                                                  is_training=phase)
                else:
                    pre_activation = tf.add(tf.matmul(activation, W), b,
                                            name='logits')
                    # The last non-linearity is a softmax function
                    if last_activation is not None:
                        activation = last_activation(pre_activation, name='prediction')
                    else:
                        activation = tf.identify(pre_activation, name='prediction')
            # W_flattened = tf.reshape(W, [-1])
            # 获取 W 的值
            if flag==True:
                self.sess.run(tf.global_variables_initializer())
                W_value = self.sess.run(W)
                # 将 W 的值展平为一维数组，并输出
                W_flattened = W_value.reshape(-1)
                self.saved_weight=W_flattened
                # print("saved_weights:", self.saved_weight)  # 添加一个


        return pre_activation, activation, reg_loss

    def get_apl(self, apl):
        self.path_length = apl

def fit_tree(x_train, y_train):
    """Train decision tree to track path length."""
    # 2
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10)
    # tree = DecisionTreeClassifier()
    # y_train = np.argmax(y_train, axis=1)
    tree.fit(x_train, y_train)
    return tree

    # 平均路径长度（计算向所提供的决策树中的输出节点进行特定输入所需的节点数。）

def average_length(tree, X):
    """Compute average path length: cost of simulating the average
    example; this is used in the objective function.

    @param tree: DecisionTreeClassifier instance
    @param X: NumPy array (D x N)
              D := number of dimensions
              N := number of examples
    @return path_length: float
                         average path length
    """
    # 获取输入样本所在的叶子节点的索引
    leaf_indices = tree.apply(X)
    # 统计叶子节点索引数组中每个元素出现的次数
    leaf_counts = np.bincount(leaf_indices)
    # tree_.node_count表示节点数量, leaf_i即为一个长度为节点数量的一维数组，其中每个元素从0开始依次递增
    leaf_i = np.arange(tree.tree_.node_count)
    path_length = np.dot(leaf_i, leaf_counts) / float(X.shape[0])
    return path_length

def average_path_length(x_train, y_train):

    tree = fit_tree(x_train, y_train)
    path_length =average_length(tree, x_train)
    return path_length,tree
def average_path_length_batch(x_train, y_train):
    num = len(y_train)
    apl_batch = np.zeros((num, 1))
    for i in range(num):
        apl_batch[i, 0] = average_path_length(x_train, y_train)
    return apl_batch
class MLPRegressor:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, learning_rate=0.01):
        # Set random seed for reproducibility
        tf.set_random_seed(42)

        # Define MLP architecture
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output

        # Define placeholders for input and output
        self.x = tf.placeholder(tf.float32, [None, n_input], name='x',dtype=tf.float32)
        self.y = tf.placeholder(tf.float32, [None, n_output], name='y',dtype=tf.float32)

        # Define weights and biases for the MLP layers
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden1]), name='W1'),
            'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name='W2'),
            'out': tf.Variable(tf.random_normal([n_hidden2, n_output]), name='W3')
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden1]), name='b1'),
            'b2': tf.Variable(tf.random_normal([n_hidden2]), name='b2'),
            'out': tf.Variable(tf.random_normal([n_output]), name='b3')
        }

        # Define MLP computation graph
        layer1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, self.weights['h2']), self.biases['b2'])
        layer2 = tf.nn.relu(layer2)

        self.prediction = tf.add(tf.matmul(layer2, self.weights['out']), self.biases['out'])

        # Define loss function and optimizer
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Create saver to save and load model parameters
        self.saver = tf.train.Saver()

        # Create session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, x_train, y_train, epochs=1000, batch_size=32, verbose=True):
        num_batches = int(len(x_train) / batch_size)

        for epoch in range(epochs):
            epoch_loss = 0

            # Shuffle data
            permutation = np.random.permutation(len(x_train))
            shuffled_x = x_train[permutation]
            shuffled_y = y_train[permutation]

            # Train on batches
            for i in range(num_batches):
                batch_x = shuffled_x[i*batch_size:(i+1)*batch_size]
                batch_y = shuffled_y[i*batch_size:(i+1)*batch_size]

                # Run optimizer and compute loss
                _, batch_loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: batch_x, self.y: batch_y})
                epoch_loss += batch_loss

            # Print epoch loss
            if verbose:
                print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss/num_batches))

        # Save model parameters
        self.saver.save(self.sess, './saved_model')

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.x: x})



from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
def avg_path_length(X, y):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=2)
    tree.fit(X, y)

    # 现在导出决策树结构,得到根节点
    dot_data = export_graphviz(tree, out_file=None)
    root_node = next(node for node in dot_data.splitlines() if node.startswith("0 -> "))

    path_lengths = []
    for sample in X:
        path_length = 0
        node = root_node
        while True:
            if node.endswith("leaf"):
                break
            path_length += 1
            node = next(node for node in dot_data.splitlines()
                        if node.startswith(f"{path_length} -> ")
                        and sample[int(node.split("<= ")[1].split()[0])]
                        <= float(node.split("<= ")[1].split()[2]))
        path_lengths.append(path_length)

    return np.mean(path_lengths)



