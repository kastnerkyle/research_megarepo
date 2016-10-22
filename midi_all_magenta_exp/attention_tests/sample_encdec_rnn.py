from __future__ import print_function
import tensorflow as tf
from tfkdllib import numpy_softmax, numpy_sample_softmax

def validate_sample_args(save_dir,
                         prime,
                         sample, **kwargs):
    return (save_dir, prime, sample)


def sample(kwargs):
    (save_dir,
     prime,
     sample) = validate_sample_args(**kwargs)
    # Wow this is nastyyyyy
    from basic_encdec import *

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        i_enc_h1 = np.zeros((batch_size, enc_h1_dim)).astype("float32")
        strings = ["hello", "please", "reverse", "me"]
        maxlen = max([len(s) for s in strings])
        inputs = np.zeros((maxlen, batch_size, 1)).astype("float32")
        inputs_mask = np.zeros_like(inputs[:, :, 0])
        targets = np.zeros((maxlen + 1, batch_size, 1)).astype("float32")
        for i, s in enumerate(strings):
            inputs_mask[:len(s), i] = 1.
            ss = [ord(si) - 31 for si in s]
            inputs[:len(ss), i, 0] = np.array(ss)
            targets[0, i, 0] = np.array(ss)[-1]
        targets_mask = inputs_mask
        targets_mask[-1, :] = 1
        targets_mask = np.roll(targets_mask, 1, axis=0)

        for j in range(1, maxlen + 1):
            random_state = np.random.RandomState(1999)

            print("Sample %i" % j)
            feed = {inpt: inputs,
                    inpt_mask: inputs_mask,
                    target: targets[:j],
                    target_mask: targets_mask[:j],
                    init_enc_h1: i_enc_h1}
            outs = [pred,]
            this_pred, = sess.run(outs, feed)
            this_softmax = numpy_softmax(this_pred)
            this_sample = numpy_sample_softmax(this_softmax, random_state)
            targets[j] = this_sample[j - 1, :, None]
        from IPython import embed; embed()
        raise ValueError()


if __name__ == '__main__':
    # prime is the text to prime with
    # sample is 0 for argmax, 1 for sample per character, 2 to sample per space
    kwargs = {"save_dir": "save/",
              "prime": 127,
              "sample": 1}
    sample(kwargs)
