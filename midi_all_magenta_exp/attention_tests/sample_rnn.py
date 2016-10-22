from __future__ import print_function
import tensorflow as tf
import os
from tfkdllib import numpy_softmax, numpy_sample_softmax

def validate_sample_args(model_ckpt,
                         prime,
                         sample,
                         sample_len,
                         **kwargs):
    return (model_ckpt, prime, sample, sample_len)


def sample(kwargs):
    (model_ckpt,
     prime,
     sample,
     sample_len) = validate_sample_args(**kwargs)
    # Wow this is nastyyyyy
    from basic_rnn import *

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        model_dir = str(os.sep).join(model_ckpt.split(os.sep)[:-1])
        model_name = model_ckpt.split(os.sep)[-1]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("Unable to restore from checkpoint")
        i_h1 = np.zeros((batch_size, h1_dim)).astype("float32")
        i_h2 = np.zeros((batch_size, h2_dim)).astype("float32")
        i_h3 = np.zeros((batch_size, h3_dim)).astype("float32")
        # start with 2 due to negative index edge case
        inputs = train_itr.transform([" "] * batch_size)

        shp = inputs.shape
        full = np.zeros((sample_len, shp[1], shp[2]), dtype="float32")
        full[:len(inputs)] = inputs[:]

        for j in range(len(inputs), sample_len):
            random_state = np.random.RandomState(1999)
            feed = {inpt: inputs,
                    init_h1: i_h1,
                    init_h2: i_h2,
                    init_h3: i_h3}
            outs = [pred, final_h1, final_h2, final_h3]
            this_pred, h1_l, h2_l, h3_l = sess.run(outs, feed)
            this_softmax = numpy_softmax(this_pred)
            this_sample = numpy_sample_softmax(this_softmax, random_state)
            full[j, :, 0] = this_sample[-1]
            # priming sequence
            inputs = this_sample[-1:, :, None]
            i_h1 = h1_l
            i_h2 = h2_l
            i_h3 = h3_l
        s = train_itr.inverse_transform(full)
        print(s)
        from IPython import embed; embed()
        raise ValueError()



if __name__ == '__main__':
    # prime is the text to prime with
    # sample is 0 for argmax, 1 for sample per character, 2 to sample per space
    import sys
    kwargs = {"model_ckpt": sys.argv[1],
              "prime": " ",
              "sample": 1,
              "sample_len": 200}
    sample(kwargs)
