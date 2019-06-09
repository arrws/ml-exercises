from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import collections
import time
import functools
import re

printf = functools.partial(print, flush=True)

class TextGenerator(object):

    def __init__(self, path_to_file = 'poezii.txt'):

        self.EPOCHS = 30
        # self.EPOCHS = 3

        self.BATCH_SIZE = 64
        # self.BATCH_SIZE = 4

        self.SEQ_LEN = 100
        self.steps_per_epoch = None

        self.BUFFER_SIZE = 10000    # for data shuffle space

        self.VOCAB_SIZE = None      # character vocabulary length
        self.EMBED_DIM = 256        # character embedding dimension

        self.RNN_UNITS = 1024       # RNN units number


        self.checkpoint_dir = './check2'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        self.dataset = self.get_dataset(path_to_file)


    def get_dataset(self, path_to_file):
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        print(text[:100])

        # dic = re.split('(\w+)|(\s)|(-)|(.)', ds)
        # dic = collections.Counter(ds)
        # del dic[None]
        # printf("vocabulary: ", len(ds))
        # printf(d.most_common(10))
        # printf(len(d))
        # words = list(d.keys())
        # words.sort(key = lambda x: len(x) if x else 0)
        # printf(words[:100])
        # printf(words[-20:])


        # The unique characters in the file
        vocab = sorted(set(text))
        self.VOCAB_SIZE = len(vocab)
        printf ('character vocab size', len(vocab))

        # map characters to indices
        self.char2idx = { u:i for i, u in enumerate(vocab) }
        self.idx2char = np.array(vocab)

        text_as_int = np.array([self.char2idx[c] for c in text])

        # The maximum length sentence we want for a single input in characters
        examples_per_epoch = len(text)//self.SEQ_LEN
        self.steps_per_epoch = examples_per_epoch//self.BATCH_SIZE


        # create training dataset with examples and targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


        sequences = char_dataset.batch(self.SEQ_LEN+1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        return dataset


    def build_model(self, input_size):
        # if tf.test.is_gpu_available():
        #     # rnn = tf.keras.layers.CuDNNGRU
        #     rnn = tf.keras.layers.CuDNNLSTM

        rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')
        model = tf.keras.Sequential([
                    tf.keras.layers.Embedding(self.VOCAB_SIZE,
                                              self.EMBED_DIM,
                                              batch_input_shape=[input_size, None]
                                              ),
                    rnn(self.RNN_UNITS,
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True
                        ),
                    tf.keras.layers.Dense(self.VOCAB_SIZE)
                    ])

        # rnn = functools.partial(tf.keras.layers.LSTM, recurrent_activation='sigmoid')
        # model = tf.keras.Sequential([
        #             tf.keras.layers.Embedding(self.VOCAB_SIZE,
        #                                       self.EMBED_DIM,
        #                                       batch_input_shape=[input_size, None]
        #                                       ),
        #             rnn(self.RNN_UNITS,
        #                 return_sequences = True,
        #                 recurrent_initializer = 'glorot_uniform',
        #                 stateful = True
        #                 ),
        #             # rnn(self.RNN_UNITS//2,
        #             #     return_sequences = True,
        #             #     recurrent_initializer = 'glorot_uniform',
        #             #     stateful = True
        #             #     ),
        #             tf.keras.layers.Dense(self.VOCAB_SIZE)
        #             ])

        return model



    def train_network(self):

        def kloss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)


        model = self.build_model(self.BATCH_SIZE)
        model.summary()
        # loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        model.compile(optimizer = tf.train.AdamOptimizer(), loss = kloss)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath = self.checkpoint_prefix, save_weights_only = True)
        history = model.fit(self.dataset.repeat(), epochs=self.EPOCHS, steps_per_epoch=self.steps_per_epoch, callbacks=[checkpoint_callback])
        tf.train.latest_checkpoint(self.checkpoint_dir)


        # model = self.build_model(self.BATCH_SIZE)
        # model.summary()
        # optimizer = tf.train.AdamOptimizer()

        # # Training step
        # for epoch in range(self.EPOCHS):
        #     start = time.time()

        #     # reinitializing the hidden state at the start of every epoch to None
        #     hidden = model.reset_states()

        #     for (batch_n, (inp, target)) in enumerate(self.dataset):
        #         with tf.GradientTape() as tape:
        #             # feeding the hidden state back into the model
        #             # This is the interesting step
        #             predictions = model(inp)
        #             loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)

        #         grads = tape.gradient(loss, model.trainable_variables)      # calculate gradient
        #         optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #         if batch_n % 100 == 0:
        #             printf('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch_n, loss))

        #     # saving (checkpoint) the model every 5 epochs
        #     model.save_weights(self.checkpoint_prefix.format(epoch=epoch))

        #     printf ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        #     printf ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # model.save_weights(self.checkpoint_prefix.format(epoch=epoch))


    def generate_text(self):
        model = self.build_model(input_size = 1)

        printf("Loading trainned")
        checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        # checkpoint = self.checkpoint_prefix.format(epoch = 1)
        printf(checkpoint)
        model.load_weights(checkpoint)

        model.build(tf.TensorShape([1, None]))

        model.summary()

        start_string=u"Un glas "

        # Number of characters to generate
        # num_generate = 1000
        num_generate = 200

        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 0.5


        # Here batch size == 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
            # printf(predictions, predicted_id)

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

        r = start_string + ''.join(text_generated)
        printf(r)

    def interactive_text(self):
        model = self.build_model(input_size = 1)

        printf("Loading trainned")
        checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        # checkpoint = self.checkpoint_prefix.format(epoch = 1)
        printf(checkpoint)
        model.load_weights(checkpoint)

        model.build(tf.TensorShape([1, None]))

        model.summary()

        r=u"Un glas "

        # Number of characters to generate
        # num_generate = 1000
        num_generate = 10

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 0.9

        while True:
            # Converting our start string to numbers (vectorizing)
            input_eval = [self.char2idx[s] for s in r]
            input_eval = tf.expand_dims(input_eval, 0)

            # Empty string to store our results
            text_generated = []


            # Here batch size == 1
            model.reset_states()
            for i in range(num_generate):
                predictions = model(input_eval)
                # remove the batch dimension
                predictions = tf.squeeze(predictions, 0)

                # using a multinomial distribution to predict the word returned by the model
                predictions = predictions / temperature
                predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
                # printf(predictions, predicted_id)

                # We pass the predicted word as the next input to the model
                # along with the previous hidden state
                input_eval = tf.expand_dims([predicted_id], 0)

                text_generated.append(self.idx2char[predicted_id])

            word = ''.join(text_generated).strip().split()[0]
            r = r + word + ' '

            os.system('clear')
            print(r)
            print('\n\n')
            human_word = input()
            r = r + human_word + ' '

            if len(r) - r.rfind('\n') > 30:
                r += '\n'

            os.system('clear')
            print(r)
            time.sleep(1)




p = TextGenerator(path_to_file = 'dataset.txt')
# p.train_network()
p.generate_text()
# p.interactive_text()

