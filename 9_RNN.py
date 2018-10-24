
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import math
import random
import re
from six.moves import xrange
from collections import namedtuple
from keras.preprocessing.sequence import pad_sequences

vocab_file = "aclImdb/imdb.vocab"
train_file = "aclImdb/train/labeledBow.feat"
test_file = "aclImdb/test/labeledBow.feat"
train_pos_dir = "aclImdb/train/neg/*.txt"
train_neg_dir = "aclImdb/train/pos/*.txt"
test_pos_dir = "aclImdb/test/neg/*.txt"
test_neg_dir = "aclImdb/test/pos/*.txt"
stopwords_file = "aclImdb/stopwords.txt"


# In[5]:


train_file_list = glob.glob(train_neg_dir, recursive=True) + glob.glob(train_pos_dir, recursive=True)
test_file_list = glob.glob(test_neg_dir, recursive=True) + glob.glob(test_pos_dir, recursive=True)

random.shuffle(train_file_list)
random.shuffle(test_file_list)

print("Train files number:",len(train_file_list))
test_file_list = test_file_list[:1000]
print("Test files number:",len(test_file_list))

vocab = open(vocab_file, "r").readlines()
stopwords = open(stopwords_file, "r").read()
vocab = [x for x in vocab if x not in stopwords] # remove irrelevant words
vocab_size = len(vocab)
print("Vocab size:",vocab_size)

id_to_word = {}
for i in range(vocab_size):
    id_to_word[i] = vocab[i][:-1]
word_to_id = {v: k for k, v in id_to_word.items()}

def clean(line):
    line = line.replace("<br />","")
    line = re.sub('[,.!?]', 'a', line)
    line = line.split()
    line = [x for x in line]
    return line


# In[6]:


#TEXT UTILS
current_file = 0
words_per_review = 200 # consider 200 words per review

def get_tokenized(file_list):
    global current_file
    file = file_list[current_file]
    current_file += 1

    batch_x = []
    line = [s for s in re.split("[._]",file)]
    val_y = int(line[-2])
    batch_y = [0] if val_y>5 else [1]

    lines = open(file, "r").readlines()
    for line in lines:
        line = clean(line)

        for i, word in enumerate(line):
            if word in word_to_id:
                batch_x.append(word_to_id[word])

    batch_x = np.array(pad_sequences([batch_x], maxlen = words_per_review)).reshape(words_per_review)
    batch_y = np.array(batch_y)
    return [batch_x, batch_y]


def get_batch(file_list, batch_size):
    x = []
    y = []
    for step in range(batch_size):
        batch_x, batch_y = get_tokenized(file_list)
        x.append(batch_x)
        y.append(batch_y)
    return [x,y]




# In[8]:


def lstm_cell(lstm_size, keep_prob):
    return tf.contrib.rnn.DropoutWrapper( tf.contrib.rnn.BasicLSTMCell(lstm_size), output_keep_prob=keep_prob)

def build_rnn(embed_size, batch_size, lstm_size, num_layers, dropout, learning_rate, dense_size):
    tf.reset_default_graph()

    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.int32, [None, None], name='x')
        y = tf.placeholder(tf.int32, [None, None], name='y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.name_scope("embeddings"):
        embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, x)

    with tf.name_scope("RNN"):
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size, keep_prob) for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

    with tf.name_scope("fully_connected"):
        W = tf.truncated_normal_initializer(stddev=0.1)
        b = tf.zeros_initializer()

        dense = tf.contrib.layers.fully_connected( outputs[:,-1], num_outputs = dense_size,
                                                  activation_fn = tf.sigmoid,
                                                  weights_initializer = W,
                                                  biases_initializer = b)

        dense = tf.contrib.layers.dropout( dense, keep_prob)

        h = tf.contrib.layers.fully_connected( dense, num_outputs = 1,
                                            activation_fn = tf.sigmoid,
                                            weights_initializer = W,
                                            biases_initializer = b)

    with tf.name_scope('train'):
        loss = tf.losses.mean_squared_error(y, h)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        tf.summary.scalar('loss', loss)

    with tf.name_scope("accuracy"):
        pred = tf.equal(tf.cast(tf.round(h), tf.int32), y)
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    export_nodes = ['x', 'y', 'keep_prob', 'initial_state', 'final_state','accuracy', 'h', 'loss', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    return graph


# In[12]:


embed_size = 300
batch_size = 250
lstm_size = 64
num_layers = 2
dropout = 0.5
learning_rate = 0.01
dense_size = 128

model = build_rnn(embed_size = embed_size,
                  batch_size = batch_size,
                  lstm_size = lstm_size,
                  num_layers = num_layers,
                  dropout = dropout,
                  learning_rate = learning_rate,
                  dense_size = dense_size)


# In[13]:


epochs = 3
train_iterations = len(train_file_list)//batch_size
test_iterations = len(test_file_list)//batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        print("\nEpoch: {}/{}".format(e+1, epochs))
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        state = sess.run(model.initial_state)
        current_file = 0

        for i in range(train_iterations):
            x, y = get_batch(train_file_list, batch_size)
            feed = {model.x: x, model.y: y, model.keep_prob: dropout, model.initial_state: state}

            l, a, state, _ = sess.run([model.loss, model.accuracy, model.final_state, model.optimizer], feed_dict=feed)
            train_loss.append(l)
            train_acc.append(a)

            if (i+1)%10==0:
                print("it", i+1, "loss:", "{0:.5f}".format(l), "acc:", "{0:.3f}".format(a))

        avg_train_loss = np.mean(train_loss)
        avg_train_acc = np.mean(train_acc)
        print("Train Loss: {:.3f}".format(avg_train_loss), "Train Acc: {:.3f}".format(avg_train_acc))

        state = sess.run(model.initial_state)
        current_file = 0

        for i in range(test_iterations):
            x, y = get_batch(test_file_list, batch_size)
            feed = {model.x: x, model.y: y, model.keep_prob: 1, model.initial_state: state}

            l, a, state = sess.run([model.loss, model.accuracy, model.final_state], feed_dict=feed)
            test_loss.append(l)
            test_acc.append(a)

        avg_test_loss = np.mean(test_loss)
        avg_test_acc = np.mean(test_acc)
        print("Test Loss: {:.3f}".format(avg_test_loss), "Test Acc: {:.3f}".format(avg_test_acc))


# In[ ]:



