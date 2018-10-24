
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import glob
import math
import random
import re
from six.moves import xrange

vocab_file = "aclImdb/imdb.vocab"
train_file = "aclImdb/train/labeledBow.feat"
test_file = "aclImdb/test/labeledBow.feat"
train_pos_dir = "aclImdb/train/neg/*.txt"
train_neg_dir = "aclImdb/train/pos/*.txt"
test_pos_dir = "aclImdb/test/neg/*.txt"
test_neg_dir = "aclImdb/test/pos/*.txt"
stopwords_file = "aclImdb/stopwords.txt"


# In[4]:


train_file_list = glob.glob(train_neg_dir, recursive=True) + glob.glob(train_pos_dir, recursive=True)
test_file_list = glob.glob(test_neg_dir, recursive=True) + glob.glob(test_pos_dir, recursive=True)

random.shuffle(train_file_list)
random.shuffle(test_file_list)
print("Train files number:",len(train_file_list))
test_file_list = test_file_list[:1000]
print("Test files number:",len(test_file_list))

# vocab = open(vocab_file, "r").readlines()
# vocab_size = len(vocab)
# print("Vocab size:",vocab_size)

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
    line = re.sub('[,.!?]',' ', line)
    line = line.split()
    line = [x for x in line]
    return line


# In[5]:


#TEXT UTILS
# skip = how many words to consider left & right
current_file = 0

def get_batch_for_embedding(file_list, skip=8):
    global current_file
    file = file_list[current_file]
    current_file += 1

    batch_x = []
    batch_y = []

    lines = open(file, "r").readlines()
    for line in lines:
        line = clean(line)

        for i, word in enumerate(line):
            if word in word_to_id:
                for j in range(1, skip):

                    # prev word
                    if i-j >= 0:
                        if line[i-j] in word_to_id and line[i-j] not in stopwords:
                            batch_x.append(word_to_id[word])
                            batch_y.append(word_to_id[line[i-j]])

                    # next word
                    if i+j < len(line):
                        if line[i+j] in word_to_id and line[i+j] not in stopwords:
                            batch_x.append(word_to_id[word])
                            batch_y.append(word_to_id[line[i+j]])

    batch_x = np.array(batch_x)#.reshape((len(batch_x), 1))
    batch_y = np.array(batch_y).reshape((len(batch_y), 1))
    return [batch_x, batch_y]


# In[6]:


batch_x, batch_y = get_batch_for_embedding(train_file_list)
print(batch_x.shape)
print(batch_y.shape)


# In[7]:


# EMBEDDING GRAPH

embed_size = 256    # Dimension of the embedding vector
num_sampled = 16    # Number of negative examples to sample.
emb_learning_rate = 1.0

# Validation samples: most frequent words
valid_size = 8      # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

emb_graph = tf.Graph()
with emb_graph.as_default():

    # input
    emb_x = tf.placeholder(tf.int32, shape=[None])
    emb_y = tf.placeholder(tf.int32, shape=[None, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # embeddings
    embeddings = tf.Variable( tf.random_uniform( [vocab_size, embed_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup( embeddings, emb_x)

    # variables for the NCE loss
    emb_W = tf.Variable( tf.truncated_normal( [vocab_size, embed_size], stddev=1.0/math.sqrt(embed_size)))
    emb_b = tf.Variable( tf.zeros( [vocab_size]))

    # trainning
    # avg NCE loss for a batch (automatically draws a new sample of the neg labels each time we eval the loss)
    emb_loss = tf.reduce_mean( tf.nn.nce_loss(weights=emb_W, biases=emb_b, labels=emb_y, inputs=embed, num_sampled=num_sampled, num_classes=vocab_size))
    emb_optimizer = tf.train.GradientDescentOptimizer(emb_learning_rate).minimize(emb_loss)

    # similarity
    norm = tf.sqrt( tf.reduce_sum( tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup( normalized_embeddings, valid_dataset)
    similarity = tf.matmul( valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()



# In[8]:


print("\nTRAINING EMBEDDINGS\n")
training_epochs = 4
num_steps = 25000
print_step = 1000

with tf.Session(graph=emb_graph) as session:
    init.run()
    print('Initialized')

    for epoch in range(training_epochs):
        print('\nEPOCH', epoch+1, '/', training_epochs,'\n')

        avg_loss = 0
        current_file = 0

        for step in xrange(num_steps):
            batch_x, batch_y = get_batch_for_embedding(train_file_list)
            feed_dict = {emb_x: batch_x, emb_y: batch_y}

            _, loss = session.run([emb_optimizer, emb_loss], feed_dict=feed_dict)
            if not math.isnan(loss/print_step):
                avg_loss += loss/print_step

            if (step+1) % print_step == 0:
                print('Avg loss at step', step+1, ':', avg_loss)
                avg_loss = 0

            if (step+1) % 5000 == 0:
                print("\nStep", step+1, "similarity eval:")
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = id_to_word[valid_examples[i]]
                    top_k = 6  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str=""
                    for k in xrange(top_k):
                        log_str += id_to_word[nearest[k]]+" "
                    print("Nearest to", valid_word, ":", log_str)
                print("")

    final_embeddings = normalized_embeddings.eval()


# In[9]:


# VISUALIZE EMBEDDINGS
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# def plot_with_labels(low_dim_embs, labels, filename='embed.png'):
#     assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
#     plt.figure(figsize=(18, 18))    # in inches
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)
#         plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
#     plt.savefig(filename)

# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# plot_only = 500
# low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
# labels = [id_to_word[i] for i in xrange(plot_only)]
# plot_with_labels(low_dim_embs, labels)


# In[10]:


#TEXT UTILS
words_per_review = 100
batch_size = 250
current_file = 0

stopwords = open(stopwords_file, "r").read()

def get_tokenized(file_list): # encodes a review using the words embeddings
    global current_file
    file = file_list[current_file]
    current_file += 1

    lines = open(file, "r").readlines()
    batch_x = [0] * embed_size
    num_words = 0

    for line in lines:
        for i, word in enumerate(clean(line)):
            if word in word_to_id and word not in stopwords:
                e = final_embeddings[word_to_id[word]]
                batch_x += e
                num_words += 1

    if num_words == 0:
        return [[],[]]
    batch_x = [x/num_words for x in batch_x]

    file = [s for s in re.split("[._]",file)]
    val_y = int(file[-2])
    batch_y = [0] if val_y>5 else [1]
    batch_y = np.array(batch_y)
    return [batch_x, batch_y]

def get_batch_for_classification(file_list):
    x = []
    y = []
    for step in range(batch_size):
        batch_x, batch_y = get_tokenized(file_list)
        if batch_x != []:
            x.append(batch_x)
            y.append(batch_y)
    return [np.array(x), np.array(y)]


# In[15]:


# example
batch_x, batch_y = get_batch_for_classification(train_file_list)
print("batch x format:", batch_x.shape)
print("batch y format:", batch_y.shape)
print(batch_x)


# In[18]:


# CLASSIFICATION GRAPH
cls_learning_rate = 0.05
num_labels = 1

cls_graph = tf.Graph()
with cls_graph.as_default():

    # input
    cls_x = tf.placeholder(tf.float32, [None, embed_size])
    cls_y = tf.placeholder(tf.float32, [None, num_labels])

    # variables
    cls_W = tf.Variable(tf.random_normal([embed_size, num_labels], mean=0, stddev=0.1))
    cls_b = tf.Variable(tf.random_normal([1, num_labels], mean=0, stddev=0.1))

    cls_h = tf.matmul(cls_x,cls_W) + cls_b

    cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_h, labels=cls_y))
    cls_optimizer = tf.train.AdamOptimizer(cls_learning_rate).minimize(cls_loss)

    cls_pred = tf.equal(tf.round(tf.nn.sigmoid(cls_h)), cls_y)
    cls_accuracy = tf.reduce_mean(tf.cast(cls_pred, tf.float32))

    init = tf.global_variables_initializer()


# In[ ]:


print("\nTRAINING CLASSIFICATION\n")
epochs = 4
train_iterations = len(train_file_list)//batch_size
test_iterations = len(test_file_list)//batch_size

with tf.Session(graph = cls_graph) as sess:
    sess.run(init)
    print("Initialized")

    for epoch in range(epochs):
        print("\nEpoch: {}/{}".format(epoch+1, epochs))

        current_file = 0
        avg_loss = 0.
        avg_acc = 0.

        for i in range(train_iterations):
            batch_x, batch_y = get_batch_for_classification(train_file_list)

            _,l, a = sess.run([cls_optimizer,cls_loss, cls_accuracy], feed_dict={cls_x: batch_x, cls_y: batch_y})
            avg_loss += l/train_iterations if not math.isnan(l) else 0
            avg_acc += a / train_iterations

            if (i+1)%10==0:
                print("step", i+1, "loss:", "{0:.5f}".format(l), "acc:", "{0:.5f}".format(a))

        print("Train accuracy:", avg_acc, "Average loss:", avg_loss)

        current_file = 0
        avg_acc = 0.

        for i in range(test_iterations):
            batch_x, batch_y = get_batch_for_classification(test_file_list)

            a = cls_accuracy.eval(feed_dict={cls_x: batch_x, cls_y: batch_y})
            avg_acc += a/test_iterations if not math.isnan(a) else 0

        print("Test accuracy:", avg_acc)


# In[ ]:



