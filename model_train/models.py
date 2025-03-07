import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.ops.losses.losses_impl import Reduction
from base_model import Model
from utils import *
pu_ids = [0, 1]

class Baseline(Model):
    """ Creates a deep learning model with a single convolutional layer. """
    def __init__(self, layer_sizes, optimizer, num_filters, num_features, frame_size, learning_rate, directory, second_conv=False, meta=None):
        # Initialize the Model object
        self.directory = directory
        # Create a tensorflow session for the model
        self.sess = tf.Session()
        # Define the learning rate for the neural net
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout')

        self.num_filters = num_filters
        self.num_features = num_features
        self.num_classes = 2
        self.frame_size = frame_size
        # Define placeholder for the inputs X and the labels y
        self.label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y')
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_features], name="X")
        # Create a placeholder that defines the phase: training or testing
        self.phase = tf.placeholder(tf.int32, name='phase')

        if meta is not None:
            # Load pre-trained model
            self.saver = tf.train.import_meta_graph(directory + '/' + meta)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))
            graph = tf.get_default_graph()
            conv = graph.get_tensor_by_name("conv1:0")
        else:
            conv = tf.layers.conv1d(
                inputs= tf.reshape(self.X, [-1, self.num_features, 1]),
                filters=self.num_filters,
                kernel_size=3,
                padding="same",
                strides=1,
                name='conv1',
                activation=tf.nn.relu)

            conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv1")

            self.logits, self.predictions, class_reg_loss = self.build_net(
                input=tf.reshape(conv, [-1, self.num_features*self.num_filters]),
                layer_sizes=[self.num_features*self.num_filters] + layer_sizes + [self.num_classes],
                nonlinearity=tf.nn.relu,
                scope='dnn',
                phase=self.phase)

            classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dnn")

            # Define the loss function
            cost = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.label,
                    name='loss_function')

            cost_mean = tf.reduce_mean(cost)
            cost_summary = tf.summary.scalar('cost', tf.reduce_mean(cost))

            # Define the loss function for the classifier
            self.loss = cost_mean

            # Define the train step
            self.train_step = tf.contrib.layers.optimize_loss(
                    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], name='CLASSIFIER')

            self.add_logging()  # Monitor loss, accuracy and auc
            # Initialize all the local variables (used to calculate auc & accuracy)
            local_init = tf.local_variables_initializer()
            self.sess.run(local_init)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()


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

        # Run the training step and write the results to Tensorboard

        summary, _ = self.sess.run([self.merged, self.train_step],
                                   feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))

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
        feed_dict['phase:0'] = 0
        # Run the evaluation and write the results to Tensorboard

        summary = self.sess.run(self.merged,
                                feed_dict=feed_dict)
        self.test_writer.add_summary(summary, global_step=self.sess.run(self.global_step))

    def add_logging(self, config=None):
        """ Creates summaries for accuracy, auc and loss. """

        # Calculate the accuracy for multiclass classification
        correct_prediction = tf.equal(tf.argmax(self.label, axis=1), tf.argmax(self.predictions, axis=1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        # Add the summaries
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        loss_summary = tf.summary.scalar('loss', self.loss)
        # Combine all the summaries
        self.merged = tf.summary.merge_all()
        # Create summary writers for train and test set
        self.train_writer = tf.summary.FileWriter(self.directory + '/train',
                                                  self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.directory + '/test')


class Bernoulli_Net(Model):
    """ Creates a SF estimator model with a Bernoulli distribution.
        Feature selection is applied on the features extracted by the conv layer.
    """
    def __init__(self, layer_sizes, optimizer, num_filters, num_features, num_samples, frame_size, learning_rate, feedback_distance, directory, second_conv=True, reg=1.0, meta=None,strength=80):
        # Initialize the Model object
        super().__init__(directory, learning_rate, num_filters, num_features, frame_size, num_samples,strength)

        if meta is not None:
            # Load pre-trained model
            self.saver = tf.train.import_meta_graph(directory + '/' + meta)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))
            graph = tf.get_default_graph()
            conv = graph.get_tensor_by_name("conv1:0")
        else:
            conv = tf.layers.conv1d(
                inputs= tf.reshape(self.X, [-1, self.num_features, 1]),
                filters=self.num_filters,
                kernel_size=3,
                padding="same",
                strides=1,
                name='conv1',
                activation=tf.nn.relu)

            conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv1")
            features = tf.reshape(conv, [-1, self.num_features, self.num_filters])
            flat_features = tf.reshape(conv, [-1, self.num_features*self.num_filters])
            # Build the controller to estimate the probability of each feature
            probs_logits, self.probs, policy_reg_loss = self.build_net(
                input=flat_features,
                layer_sizes=[self.num_features*self.num_filters, 128, self.num_features],
                nonlinearity=tf.nn.relu,
                last_activation=tf.nn.sigmoid,
                scope='attention',
                phase=self.phase)

            policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "attention")

            # Define a Bernoulli distribution based on the probabilities estimated by the controller
            self.distribution = tf.contrib.distributions.Bernoulli(logits=probs_logits, allow_nan_stats=False, dtype=tf.float32)
            dynamic_num_samples = tf.floordiv(tf.shape(self.attention)[0], tf.shape(self.X)[0])
            # Calculate the probability of each attention vector
            sample_probs = tf.reshape(self.distribution.prob(tf.reshape(self.attention, [dynamic_num_samples, -1, self.num_features])), [-1, self.num_features])

            # Define summaries to monitor the agent
            attention_summary = tf.summary.histogram('a', self.attention)

            probability_summary = tf.summary.histogram('probability', self.probs)
            sample_probability_summary = tf.summary.histogram('sample_probability', sample_probs)

            # Calculate regularizers based on the sparsity of the distribution of probabilities
            batch_mean, batch_variance = tf.nn.moments(self.probs, axes=[0])
            example_mean, example_variance = tf.nn.moments(self.probs, axes=[1])
            Lb = tf.reduce_mean(tf.square(batch_mean - 0.20))
            Le = tf.reduce_mean(tf.square(example_mean - 0.20))
            Lve = tf.reduce_mean(example_variance)
            Lvb = tf.reduce_mean(batch_variance)
            self.reg_loss = 5 * (Le + Lb) - (Lve + Lvb)

            Lb_summary = tf.summary.scalar('Lb', Lb)
            Le_summary = tf.summary.scalar('Le', Le)
            Lv_summary = tf.summary.scalar('Lve', Lve)
            Lv_summary = tf.summary.scalar('Lvb', Lvb)

            # Multiply the features by the gates defined by attention
            features = tf.reshape(tf.tile(tf.reshape(features, [-1]), [dynamic_num_samples]), [-1, self.num_features, self.num_filters])
            self.attention = tf.cond(tf.greater(self.phase, 0), lambda: self.attention, lambda: tf.round(self.attention))
            zoom_X = tf.multiply(features, tf.reshape(self.attention, [-1, self.num_features, 1]), name='Zoom_X')

            if second_conv:
                # Multiply the features by the gates defined by attention
                zoom_X = tf.layers.conv1d(
                            inputs= tf.reshape(zoom_X,[-1, self.num_features, 1]),
                            filters=self.num_filters,
                            # filters=1,
                            kernel_size=3,
                            padding="same",
                            strides=1,
                            name="conv2",
                            activation=tf.nn.relu)
                            # kernel_regularizer=l2_regularizer,
                            # bias_regularizer=l2_regularizer)


                conv_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv2")
                zoom_size = self.num_features*self.num_filters
            else:
                conv_vars2 = []
                zoom_size = self.num_features*self.num_filters

            flat_zoom_X = tf.reshape(zoom_X, [-1, zoom_size])

            self.logits, self.predictions, class_reg_loss = self.build_net(
                input=flat_zoom_X,
                layer_sizes=[zoom_size] + layer_sizes + [self.num_classes],
                nonlinearity=tf.nn.relu,
                scope='dnn',
                phase=self,
                flag=True)

            classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dnn")
            # 打印变量的值
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     classifier_vars_val = sess.run(classifier_vars)
            #     print('classifier_vars')
            #     print(classifier_vars_val)
            # Define the loss function
            cost = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=tf.reshape(tf.tile(tf.reshape(self.label, [-1]),
                     [dynamic_num_samples]), [-1, self.num_classes]),
                    name='loss_function')



            past_cost = tf.Variable(tf.zeros([]), name='past_cost')

            cost_mean = tf.reduce_mean(cost)
            cost_summary = tf.summary.scalar('cost', tf.reduce_mean(cost))

            log_probability = tf.log(sample_probs + 10e-20)
            log_probability = tf.verify_tensor_all_finite(log_probability, "Undefined log probability")
            log_probability = tf.reduce_sum(log_probability, axis=1)

            # Define the loss function for the classifier
            self.loss = cost_mean
            self.policy_loss = tf.reduce_mean(tf.multiply((cost-past_cost), log_probability))
            self.reg_loss = 10*(Le + Lb) - (Lve + Lvb)

            self.pre_train_step = tf.contrib.layers.optimize_loss(
                    cost_mean, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=[classifier_vars, conv_vars, conv_vars2], name='PRE_TRAIN')

            # Define the train step
            self.train_step = [
                tf.contrib.layers.optimize_loss(
                    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=[classifier_vars, conv_vars2], name='CLASSIFIER'),
                tf.contrib.layers.optimize_loss(
                    self.policy_loss + reg*self.reg_loss, global_step=self.global_step, increment_global_step=False, learning_rate=0.05*self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=policy_vars, name='REINFORCE')
            ]

            if feedback_distance == 'cosine':
                feedback_cost = tf.losses.cosine_distance(
                        labels=self.feedback,
                        predictions=tf.reshape(self.probs, [-1, self.num_features]),
                        dim=1,
                        reduction=Reduction.NONE)
            elif feedback_distance == 'mse':
                feedback_cost = tf.losses.mean_squared_error(
                        labels=self.feedback,
                        predictions=tf.reshape(self.probs, [-1, self.num_features]),
                        reduction=Reduction.NONE)
            elif feedback_distance == 'log_loss':
                feedback_cost = tf.losses.log_loss(
                        labels=self.feedback,
                        predictions=tf.reshape(self.probs, [-1, self.num_features]),
                        reduction=Reduction.NONE)
            self.cnn_tree_train_step = tf.contrib.layers.optimize_loss(
                self.loss + self.strength * self.path_length,
                global_step=self.global_step, learning_rate=0.1 * self.learning_rate, optimizer=optimizer,
                summaries=["gradients"], variables=[classifier_vars,policy_vars,], name='CNN_TREE_TRAIN')

            self.feedback_train_step = tf.contrib.layers.optimize_loss(
                tf.reduce_mean(feedback_cost) + self.policy_loss +self.reg_loss,
                global_step=self.global_step, learning_rate=0.1*self.learning_rate, optimizer=optimizer,
                summaries=["gradients"], variables=[policy_vars,conv_vars], name='FEEDBACK_TRAIN')
            # self.loss + tf.reduce_mean(feedback_cost) + self.strength * self.path_length
            self.feedback_tree_train_step = tf.contrib.layers.optimize_loss(
                 self.loss+tf.reduce_mean(feedback_cost)+self.strength*self.path_length,
                global_step=self.global_step, learning_rate=0.1 * self.learning_rate, optimizer=optimizer,
                summaries=["gradients"], variables=[classifier_vars], name='FEEDBACK_TREE_TRAIN')
            # classifier_vars, policy_vars, conv_vars

            tf.assign(past_cost, past_cost*0.9 + 0.1*tf.reduce_mean(cost))

            self.add_logging()  # Monitor loss, accuracy and auc
            # Initialize all the local variables (used to calculate auc & accuracy)
            local_init = tf.local_variables_initializer()
            self.sess.run(local_init)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

            # input_images = tf.summary.image("Visualize_Input", tf.reshape(self.X, [10, frame_size, frame_size, 1]), max_outputs=10)
            # prob_images = tf.summary.image("Probability_Distribution", tf.reshape(self.probs, [10, int(frame_size/2), int(frame_size/2), 1]), max_outputs=10)
            # feedback_images = tf.summary.image("Feedback_Images", tf.reshape(self.feedback, [10, int(frame_size/2), int(frame_size/2), 1]), max_outputs=10)
            # attention_images = tf.summary.image("Attention_Images", tf.reshape(self.attention, [10, int(frame_size/2), int(frame_size/2), 1]), max_outputs=10)
            # self.merged_images = tf.summary.merge([input_images, prob_images, attention_images, feedback_images])



