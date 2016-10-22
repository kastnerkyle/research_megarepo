from __future__ import print_function
import tensorflow as tf
import os
from tfkdllib import numpy_softmax, numpy_sample_softmax, piano_roll_to_midi
from tfkdllib import notes_to_midi


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
    from roll_rnn import *
    all_notes = []
    for i in range(10):
        notes_mb = train_itr.next()
        all_notes.append(notes_mb[:, 0])
    train_itr.reset()
    all_notes = np.concatenate(all_notes, axis=0)
    notes_to_midi("gt.mid", all_notes)

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
        i_h1 = np.zeros((batch_size, h_dim)).astype("float32")
        i_h2 = np.zeros((batch_size, h_dim)).astype("float32")
        note_inputs = np.zeros((1, batch_size, train_itr.simultaneous_notes))
        note_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))

        shp = note_inputs.shape
        full_notes = np.zeros((sample_len, shp[1], shp[2]), dtype="float32")
        full_notes[:len(note_inputs)] = note_inputs[:]

        random_state = np.random.RandomState(1999)
        for j in range(len(note_inputs), sample_len):
            for ni in range(n_notes):
                feed = {note_inpt: note_inputs,
                        note_target: note_targets,
                        init_h1: i_h1,
                        init_h2: i_h2}
                outs = []
                outs += note_preds
                outs += [final_h1, final_h2]
                r = sess.run(outs, feed)
                h_l = r[-2:]
                h1_l, h2_l = h_l
                this_preds = r[:-2]
                this_probs = [numpy_softmax(p) for p in this_preds]
                # reweight for no silence, ever?
                this_samples = [numpy_sample_softmax(p, random_state)
                                for p in this_probs]
                note_probs = this_probs[:n_notes]
                # only put the single note in...
                full_notes[j, :, ni] = this_samples[ni].ravel()
                note_targets[0, :, ni] = this_samples[ni].ravel()
            # priming sequence
            note_inputs = full_notes[j:j+1]
            note_targets = np.zeros((1, batch_size, train_itr.simultaneous_notes))
            i_h1 = h1_l
            i_h2 = h2_l
        notes_to_midi("temp.mid", full_notes)


if __name__ == '__main__':
    # prime is the text to prime with
    # sample is 0 for argmax, 1 for sample per character, 2 to sample per space
    import sys
    kwargs = {"model_ckpt": sys.argv[1],
              "prime": " ",
              "sample": 1,
              "sample_len": 1000}
    sample(kwargs)
