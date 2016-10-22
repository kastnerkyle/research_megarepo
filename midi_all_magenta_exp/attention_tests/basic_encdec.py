from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from tfkdllib import ni, scan, broadcast
from tfkdllib import shift
from tfkdllib import print_network
from tfkdllib import OneHot, GRUFork, GRU, Linear
from tfkdllib import tanh
from tfkdllib import softmax, categorical_crossentropy
from tfkdllib import character_file_iterator

batch_size = 64
train_itr = character_file_iterator("data/dictionary_shuf.txt", batch_size,
                                    make_mask=True)
valid_itr = character_file_iterator("data/dictionary_shuf.txt", batch_size,
                                    make_mask=True)
mb, mb_mask = next(train_itr)
train_itr.reset()

checkpoint_path = "save/encdec_model.ckpt"
num_features = mb.shape[-1]
n_symbols = 95
num_epochs = 1000
n_dim = 512
embed1_dim = 100
enc_h1_dim = n_dim
dec_h1_dim = n_dim
out_dim = n_symbols
learning_rate = .0001
random_state = np.random.RandomState(1999)

inpt = tf.placeholder(tf.float32, [None, batch_size, num_features])
target = tf.placeholder(tf.float32, [None, batch_size, num_features])

inpt_mask = tf.placeholder(tf.float32, [None, batch_size])
target_mask = tf.placeholder(tf.float32, [None, batch_size])

init_enc_h1 = tf.placeholder(tf.float32, [batch_size, enc_h1_dim])

oh_i = OneHot(inpt, n_symbols)
inp_proj, inpgate_proj = GRUFork([oh_i], [n_symbols], enc_h1_dim, random_state)


def enc_step(inp_t, inpgate_t, inpmask_t, h1_tm1):
    enc_h1 = GRU(inp_t, inpgate_t, h1_tm1, enc_h1_dim, enc_h1_dim, random_state,
                 mask=inpmask_t)
    return enc_h1

enc_h1 = scan(enc_step, [inp_proj, inpgate_proj, inpt_mask], [init_enc_h1])
final_enc_h1 = ni(enc_h1, -1)

# Kick off dynamics
init_dec_h1 = tanh(Linear([final_enc_h1], [enc_h1_dim], dec_h1_dim,
                          random_state))
oh_target = OneHot(target, n_symbols)

# prepend 0, then slice off last timestep
shift_target = shift(oh_target)
# shift mask the same way? but use 1 to mark as active
# shift_target_mask = shift(target_mask, fill_value=1.)

out_proj, outgate_proj = GRUFork([shift_target], [n_symbols],
                                 dec_h1_dim, random_state)

# Just add in at each timestep - no easy broadcast target here without some work
outctx_proj, outctxgate_proj = GRUFork([final_enc_h1], [enc_h1_dim],
                                       dec_h1_dim, random_state)


def dec_step(out_t, outgate_t, outmask_t, h1_tm1):
    dec_h1 = GRU(out_t + outctx_proj, outgate_t + outctxgate_proj,
                 h1_tm1, dec_h1_dim, dec_h1_dim, random_state,
                 mask=outmask_t)
    return dec_h1

dec_h1 = scan(dec_step, [out_proj, outgate_proj, target_mask],
              [init_dec_h1])

# Add decode context with shape/broadcast games
ctx = broadcast(final_enc_h1, dec_h1)
pred = Linear([dec_h1, ctx], [dec_h1_dim, enc_h1_dim], out_dim, random_state)

full_cost = categorical_crossentropy(softmax(pred), target)
cost = tf.reduce_sum(target_mask * full_cost) / batch_size

# cost in bits
# cost = cost * 1.44269504089
params = tf.trainable_variables()
print_network(params)
grads = tf.gradients(cost, params)
grad_clip = 5.0
grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
opt = tf.train.AdamOptimizer(learning_rate)
updates = opt.apply_gradients(zip(grads, params))


def _loop(X_mb, X_mb_mask, y_mb, y_mb_mask, do_updates=True):
    i_enc_h1 = np.zeros((batch_size, enc_h1_dim)).astype("float32")
    i_dec_h1 = np.zeros((batch_size, dec_h1_dim)).astype("float32")
    costs = []
    X_sub = X_mb.astype("float32")
    y_sub = y_mb.astype("float32")
    feed = {inpt: X_sub,
            inpt_mask: X_mb_mask,
            target: y_sub,
            target_mask: y_mb_mask,
            init_enc_h1: i_enc_h1,
            init_dec_h1: i_dec_h1}
    if do_updates:
        outs = [cost, final_enc_h1, updates]
        train_loss, enc_h1_l, _ = sess.run(outs, feed)
    else:
        outs = [cost, final_enc_h1]
        train_loss, enc_h1_l = sess.run(outs, feed)
    i_enc_h1 = enc_h1_l
    costs.append(train_loss)
    return costs


if __name__ == "__main__":
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        if os.path.exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)

        for e in range(num_epochs):
            saver.save(sess, checkpoint_path)
            # train_loop
            train_itr.reset()
            print("Epoch %i" % e)
            i = 0
            print("Train")
            window_size = ws = 20.
            running_mean = 0.
            overall_mean = 0.
            start = time.time()
            for full in train_itr:
                i += 1
                mb, mb_mask = full
                X_mb = mb[:-1]
                X_mb_mask = mb_mask[:-1]
                y_mb = mb[1:]
                y_mb_mask = mb_mask[1:]
                train_costs = _loop(X_mb, X_mb_mask, y_mb, y_mb_mask)
                if (i % 100 == 0) and i > 0:
                    print("At %i, recent mean: %f" % (i, running_mean))
                # should only have 1 element?
                c = np.sum(train_costs)
                # train_costs
                overall_mean = overall_mean + (c - overall_mean) / float(i)
                running_mean = running_mean + (c - running_mean) / float(ws)
            print("Mean recent cost: %f" % running_mean)
            print("Overall mean cost: %f" % overall_mean)
            end = time.time()
            print("Time: %f" % (end - start))
