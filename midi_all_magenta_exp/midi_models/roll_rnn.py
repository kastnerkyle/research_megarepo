from __future__ import print_function
import tensorflow as tf
import numpy as np
from tfkdllib import ni, scan
from tfkdllib import Multiembedding, GRUFork, GRU, Linear, Automask
from tfkdllib import softmax, categorical_crossentropy
from tfkdllib import run_loop
from tfkdllib import tfrecord_roll_iterator


batch_size = 50
sequence_length = 20

train_itr = tfrecord_roll_iterator("BachChorales.tfrecord",
                                   batch_size,
                                   stop_index=.9,
                                   sequence_length=sequence_length)

valid_itr = tfrecord_roll_iterator("BachChorales.tfrecord",
                                   batch_size,
                                   start_index=.9,
                                   sequence_length=sequence_length)
note_mb = next(train_itr)
train_itr.reset()

num_note_features = note_mb.shape[-1]
num_epochs = 100
n_note_symbols = len(train_itr.note_classes)
n_notes = train_itr.simultaneous_notes
note_embed_dim = 32
n_dim = 256
h_dim = n_dim
note_out_dims = n_notes * [n_note_symbols]

learning_rate = .001
grad_clip = 5.0
random_state = np.random.RandomState(1999)

note_inpt = tf.placeholder(tf.float32, [None, batch_size, num_note_features])
note_target = tf.placeholder(tf.float32, [None, batch_size, num_note_features])
init_h1 = tf.placeholder(tf.float32, [batch_size, h_dim])
init_h2 = tf.placeholder(tf.float32, [batch_size, h_dim])

note_embed = Multiembedding(note_inpt, n_note_symbols, note_embed_dim, random_state)
inp_proj, inpgate_proj = GRUFork([note_embed],
                                 [n_notes * note_embed_dim],
                                 h_dim,
                                 random_state)

def step(inp_t, inpgate_t, h1_tm1, h2_tm1):
    h1 = GRU(inp_t, inpgate_t, h1_tm1, h_dim, h_dim, random_state)
    h2 = GRU(inp_t, inpgate_t, h2_tm1, h_dim, h_dim, random_state)
    return h1, h2

h1, h2 = scan(step, [inp_proj, inpgate_proj], [init_h1, init_h2])
final_h1 = ni(h1, -1)
final_h2 = ni(h2, -1)

target_note_embed = Multiembedding(note_target, n_note_symbols, note_embed_dim,
                                   random_state)
target_note_masked = Automask(target_note_embed, n_notes)

costs = []
note_preds = []
duration_preds = []
for i in range(n_notes):
    note_pred = Linear([h1, h2, target_note_masked[i]],
                       [h_dim, h_dim, n_notes * note_embed_dim],
                       note_out_dims[i], random_state, weight_norm=False)
    # reweight by empirical counts?
    n = categorical_crossentropy(softmax(note_pred), note_target[:, :, i],
                                 class_weights={0: .001})
    cost = tf.reduce_sum(n)
    note_preds.append(note_pred)
    costs.append(cost)

cost = sum(costs) #/ (sequence_length * batch_size)

# cost in bits
# cost = cost * 1.44269504089
params = tf.trainable_variables()
grads = tf.gradients(cost, params)
grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
opt = tf.train.AdamOptimizer(learning_rate)
updates = opt.apply_gradients(zip(grads, params))


def _loop(itr, sess, inits=None, do_updates=True):
    if inits is None:
        i_h1 = np.zeros((batch_size, h_dim)).astype("float32")
        i_h2 = np.zeros((batch_size, h_dim)).astype("float32")
    else:
        i_h1, i_h2 = inits
    note_mb = next(itr)
    X_note_mb = note_mb[:-1]
    y_note_mb = note_mb[1:]
    feed = {note_inpt: X_note_mb,
            note_target: y_note_mb,
            init_h1: i_h1,
            init_h2: i_h2}
    if do_updates:
        outs = [cost, final_h1, final_h2, updates]
        train_loss, h1_l, h2_l, _ = sess.run(outs, feed)
    else:
        outs = [cost, final_h1, final_h2]
        train_loss, h1_l, h2_l = sess.run(outs, feed)
    return train_loss, h1_l, h2_l


if __name__ == "__main__":
    run_loop(_loop, train_itr, valid_itr,
             n_epochs=num_epochs,
             checkpoint_delay=10,
             checkpoint_every_n_epochs=5)
