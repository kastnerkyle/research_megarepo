from __future__ import print_function
import tensorflow as tf
import numpy as np
from tfkdllib import ni, scan
from tfkdllib import print_network
from tfkdllib import Embedding, GRUFork, GRU, Linear
from tfkdllib import softmax, categorical_crossentropy
from tfkdllib import character_file_iterator
from tfkdllib import run_loop

batch_size = 50
sequence_length = 50
train_itr = character_file_iterator("data/shakespeare.txt", batch_size,
                                    make_mask=True,
                                    sequence_length=sequence_length)
valid_itr = character_file_iterator("data/shakespeare.txt", batch_size,
                                    make_mask=True,
                                    sequence_length=sequence_length,
                                    stop_index=1000 * sequence_length)
mb, mb_mask = next(train_itr)
train_itr.reset()

num_features = mb.shape[-1]
n_symbols = train_itr.vocabulary_size
num_epochs = 50
n_dim = 128
embed1_dim = 128
h1_dim = n_dim
h2_dim = n_dim
h3_dim = n_dim
out_dim = n_symbols
learning_rate = .01
grad_clip = 5.0
random_state = np.random.RandomState(1999)

inpt = tf.placeholder(tf.float32, [None, batch_size, num_features])
target = tf.placeholder(tf.float32, [None, batch_size, num_features])
init_h1 = tf.placeholder(tf.float32, [batch_size, h1_dim])
init_h2 = tf.placeholder(tf.float32, [batch_size, h2_dim])
init_h3 = tf.placeholder(tf.float32, [batch_size, h3_dim])

embed1 = Embedding(inpt, n_symbols, embed1_dim, random_state)
inp_proj, inpgate_proj = GRUFork([embed1], [embed1_dim], h1_dim, random_state)


def step(inp_t, inpgate_t, h1_tm1, h2_tm1, h3_tm1):
    h1 = GRU(inp_t, inpgate_t, h1_tm1, h1_dim, h1_dim, random_state)
    h1_t, h1gate_t = GRUFork([h1], [h1_dim], h2_dim, random_state)
    h2 = GRU(h1_t, h1gate_t, h2_tm1, h2_dim, h2_dim, random_state)
    h2_t, h2gate_t = GRUFork([h2], [h2_dim], h3_dim, random_state)
    h3 = GRU(h2_t, h2gate_t, h3_tm1, h3_dim, h3_dim, random_state)
    return h1, h2, h3

h1, h2, h3 = scan(step, [inp_proj, inpgate_proj], [init_h1, init_h2, init_h3])
final_h1, final_h2, final_h3 = [ni(h1, -1), ni(h2, -1), ni(h3, -1)]

pred = Linear([h3], [h3_dim], out_dim, random_state)
cost = tf.reduce_mean(categorical_crossentropy(softmax(pred), target))

# cost in bits
# cost = cost * 1.44269504089
params = tf.trainable_variables()
print_network(params)
grads = tf.gradients(cost, params)
grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
opt = tf.train.AdamOptimizer(learning_rate)
updates = opt.apply_gradients(zip(grads, params))


def _loop(itr, sess, inits=None, do_updates=True):
    if inits is None:
        i_h1 = np.zeros((batch_size, h1_dim)).astype("float32")
        i_h2 = np.zeros((batch_size, h2_dim)).astype("float32")
        i_h3 = np.zeros((batch_size, h3_dim)).astype("float32")
    else:
        i_h1, i_h2, i_h3 = inits
    mb, mb_mask = next(itr)
    X_mb = mb[:-1]
    y_mb = mb[1:]
    X_sub = X_mb.astype("float32")
    y_sub = y_mb.astype("float32")
    feed = {inpt: X_sub,
            target: y_sub,
            init_h1: i_h1,
            init_h2: i_h2,
            init_h3: i_h3}
    if do_updates:
        outs = [cost, final_h1, final_h2, final_h3, updates]
        train_loss, h1_l, h2_l, h3_l, _ = sess.run(outs, feed)
    else:
        outs = [cost, final_h1, final_h2, final_h3]
        train_loss, h1_l, h2_l, h3_l = sess.run(outs, feed)
    return train_loss, h1_l, h2_l, h3_l


if __name__ == "__main__":
    run_loop(_loop, train_itr, valid_itr,
             n_epochs=num_epochs,
             checkpoint_delay=10,
             checkpoint_every_n_epochs=1)
