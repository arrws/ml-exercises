
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import random
import re
from sklearn.decomposition import PCA

logs_path = "/home/ape/Jupyter/logs"
vocab_file = "aclImdb/imdb.vocab"
train_file = "aclImdb/train/labeledBow.feat"
test_file = "aclImdb/test/labeledBow.feat"


# In[2]:


# text utils
train_lines = []
train_lines = open(train_file, "r").readlines()
random.shuffle(train_lines)
print("train size:", len(train_lines))

test_lines = []
test_lines = open(test_file, "r").readlines()
random.shuffle(test_lines)
test_lines = test_lines[:1000]
print("test size:", len(test_lines))

vocab = open(vocab_file, "r").readlines()
vocab_size = len(vocab)
print("Vocab size:",vocab_size)

current_line = 0

def get_batch(lines, batch_size):
    global current_line
    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        x = np.zeros(vocab_size, np.int32)
        line = [int(s) for s in re.split("[ :\n]",lines[current_line]) if s.isdigit()]
        current_line += 1

        for k in range(1,len(line)-1,2):
            x[line[k]] = 1 # doc2vec

        batch_x.append(x)
        batch_y.append([0] if line[0]>5 else [1])

    return [np.asarray(batch_x), np.asarray(batch_y)]


# In[3]:


# pca
no_comp = 500
x, _ = get_batch(train_lines, no_comp)
pca = PCA(n_components=no_comp)
pca.fit(x)


# In[4]:

learning_rate = 0.01
num_labels = 1

# Model

x = tf.placeholder(tf.float32, [None, no_comp], name = 'x')
y = tf.placeholder(tf.int32, [None, num_labels], name = 'y')

W = tf.Variable(tf.random_normal([no_comp, num_labels], mean=0, stddev=0.1), name="W")
b = tf.Variable(tf.random_normal([1, num_labels], mean=0, stddev=0.1), name="b")

h = tf.nn.sigmoid(tf.matmul(x, W) + b)

loss = cost = tf.losses.mean_squared_error(y, h)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

pred = tf.equal(tf.cast(tf.round(h), tf.int32), y)
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged = tf.summary.merge_all()


# In[5]:


epochs = 3
batch_size = 250
train_iterations = len(train_lines)//batch_size
test_iterations = len(test_lines)//batch_size

log_period = 10

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path, sess.graph)
    print("Initialized")

    for epoch in range(epochs):
        print("\nEpoch: {}/{}".format(epoch+1, epochs))
        current_line = 0
        avg_loss = 0.
        avg_acc = 0.

        for i in range(train_iterations):
            batch_x, batch_y = get_batch(train_lines, batch_size)
            batch_x = pca.transform(batch_x)

            _, l, a, s = sess.run([optimizer, loss, accuracy, merged], feed_dict={x: batch_x, y: batch_y})

            avg_loss += l / train_iterations
            avg_acc += a / train_iterations

            writer.add_summary(s, i+epoch*train_iterations)

            if (i+1)%10==0:
                print("step", i+1, "loss:", "{0:.5f}".format(l), "acc:", "{0:.3f}".format(a))

        print("Train accuracy:", avg_acc, "Average loss:", avg_loss)

        current_line = 0
        avg_acc = 0.

        for i in range(test_iterations):
            batch_x, batch_y = get_batch(test_lines, batch_size)
            batch_x = pca.transform(batch_x)

            a = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
            avg_acc += a / test_iterations

        print("Test accuracy:", avg_acc)

    writer.close()



# In[ ]:



