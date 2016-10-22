from __future__ import print_function
import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import audio_file_iterator
from kdllib import numpy_softmax, numpy_sample_softmax
from kdllib import param, param_search, print_param_info
from kdllib import LearnedInitHidden
from kdllib import Linear
from kdllib import Embedding
from kdllib import Igor
from kdllib import load_checkpoint, theano_one_hot, concatenate
from kdllib import fetch_fruitspeech, list_iterator
from kdllib import np_zeros, GRU, GRUFork
from kdllib import make_weights, make_biases, relu, run_loop
from kdllib import as_shared, adam, gradient_clipping
from kdllib import get_values_from_function, set_shared_variables_in_function
from kdllib import soundsc, categorical_crossentropy
from kdllib import relu, softmax, sample_softmax



if __name__ == "__main__":
    import argparse

    fs = 16000
    minibatch_size = 128
    cut_len = 64
    n_epochs = 1000  # Used way at the bottom in the training loop!
    checkpoint_every_n_epochs = 1
    checkpoint_every_n_updates = 1000
    checkpoint_every_n_seconds = 60 * 60
    random_state = np.random.RandomState(1999)

    filepath = "/Tmp/kastner/blizzard_wav_files/*flac"
    train_itr = audio_file_iterator(filepath, minibatch_size=minibatch_size,
                                    stop_index=.9, preprocess="quantize")
    valid_itr = audio_file_iterator(filepath, minibatch_size=minibatch_size,
                                    start_index=.9, preprocess="quantize")
    X_mb, X_mb_mask = next(train_itr)
    train_itr.reset()

    input_dim = 256
    n_embed = 256
    n_hid = 512
    n_proj = 512
    n_bins = 256

    desc = "Speech generation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', '--sample',
                        help='Sample from a checkpoint file',
                        default=None,
                        required=False)
    def restricted_int(x):
        if x is None:
            # None makes it "auto" sample
            return x
        x = int(x)
        if x < 1:
            raise argparse.ArgumentTypeError("%r not range [1, inf]" % (x,))
        return x
    parser.add_argument('-sl', '--sample_length',
                        help='Number of steps to sample, default is automatic',
                        type=restricted_int,
                        default=None,
                        required=False)
    def restricted_float(x):
        if x is None:
            # None makes it "auto" temperature
            return x
        x = float(x)
        if x <= 0:
            raise argparse.ArgumentTypeError("%r not range (0, inf]" % (x,))
        return x
    parser.add_argument('-t', '--temperature',
                        help='Sampling temperature for softmax',
                        type=restricted_float,
                        default=None,
                        required=False)
    parser.add_argument('-c', '--continue', dest="cont",
                        help='Continue training from another saved model',
                        default=None,
                        required=False)
    args = parser.parse_args()
    if args.sample is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        checkpoint_file = args.sample
        if not os.path.exists(checkpoint_file):
            raise ValueError("Checkpoint file path %s" % checkpoint_file,
                             " does not exist!")
        print(checkpoint_file)
        checkpoint_dict = load_checkpoint(checkpoint_file)

        X_mb, X_mb_mask = next(train_itr)
        train_itr.reset()
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]

        predict_function = checkpoint_dict["predict_function"]
        sample_function = checkpoint_dict["sample_function"]

        if args.temperature is None:
            args.temperature = 1.
        if args.sample_length is None:
            raise ValueError("NYI - use -sl or --sample_length ")
        else:
            fixed_steps = args.sample_length
            temperature = args.temperature
            completed = []
            # 0 is in the middle
            # CANNOT BE 1 timestep - will get floating point exception!
            init_x = 127 + np_zeros((2, minibatch_size, 1)).astype(theano.config.floatX)
            for i in range(fixed_steps):
                if i % 100 == 0:
                    print("Sampling step %i" % i)
                rvals = sample_function(init_x, prev_h1, prev_h2,
                                        prev_h3)
                sampled, h1_s, h2_s, h3_s = rvals
                pred_s = numpy_softmax(sampled, temperature=temperature)
                # use 0 since it is a moving window
                choice = numpy_sample_softmax(pred_s[0], random_state)
                choice = choice[None]
                completed.append(choice)

                # use 2 since scan is throwing exceptions
                init_x = np.concatenate((choice[..., None], choice[..., None]),
                                        axis=0)
                init_x = init_x.astype(theano.config.floatX)
                # use last hidden
                prev_h1 = h1_s[0]
                prev_h2 = h2_s[0]
                prev_h3 = h3_s[0]
            print("Completed sampling after %i steps" % fixed_steps)
            # mb, length
            completed = np.array(completed)[:, 0, :]
            completed = completed.transpose(1, 0)
            # all samples would be range(len(completed))
            for i in range(10):
                ex = completed[i].ravel()
                s = "gen_%i.wav" % (i)
                ex = ex.astype("float32")
                """
                ex -= ex.min()
                ex /= ex.max()
                ex -= 0.5
                ex *= 0.95
                wavfile.write(s, fs, ex)
                """
                wavfile.write(s, fs, soundsc(ex))
        print("Sampling complete, exiting...")
        sys.exit()
    else:
        print("No plotting arguments, starting training mode!")

    X_sym = tensor.tensor3("X_sym")
    X_sym.tag.test_value = X_mb[:cut_len]
    X_mask_sym = tensor.matrix("X_mask_sym")
    X_mask_sym.tag.test_value = X_mb_mask[:cut_len]

    init_h1_i = tensor.matrix("init_h1")
    init_h1_i.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h2_i = tensor.matrix("init_h2")
    init_h2_i.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h3_i = tensor.matrix("init_h3")
    init_h3_i.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h1, init_h2, init_h3 = LearnedInitHidden(
        [init_h1_i, init_h2_i, init_h3_i], 3 * [(minibatch_size, n_hid)])

    inpt = X_sym[:-1]
    target = X_sym[1:]
    mask = X_mask_sym[:-1]
    embed_dim = 256
    embed1 = Embedding(inpt, 256, embed_dim, random_state)
    in_h1, ingate_h1 = GRUFork([embed1], [embed_dim], n_hid, random_state)

    def step(in_h1_t, ingate_h1_t,
             h1_tm1, h2_tm1, h3_tm1):
        h1_t = GRU(in_h1_t, ingate_h1_t, h1_tm1, n_hid, n_hid, random_state)
        h1_h2_t, h1gate_h2_t = GRUFork([h1_t], [n_hid], n_hid, random_state)

        h2_t = GRU(h1_h2_t, h1gate_h2_t, h2_tm1,
                   n_hid, n_hid, random_state)

        h2_h3_t, h2gate_h3_t = GRUFork([h2_t], [n_hid], n_hid, random_state)

        h3_t = GRU(h2_h3_t,
                   h2gate_h3_t, h3_tm1,
                   n_hid, n_hid, random_state)
        return h1_t, h2_t, h3_t

    (h1, h2, h3), updates = theano.scan(
        fn=step,
        sequences=[in_h1, ingate_h1],
        outputs_info=[init_h1, init_h2, init_h3])
    out = Linear([h3], [n_hid], n_bins, random_state)
    pred = softmax(out)
    shp = target.shape
    target = target.reshape((shp[0], shp[1]))
    target = theano_one_hot(target, n_classes=n_bins)
    # dimshuffle so batch is on last axis
    cost = categorical_crossentropy(pred, target)
    cost = cost * mask.dimshuffle(0, 1)
    # sum over sequence length and features, mean over minibatch
    cost = cost.dimshuffle(1, 0)
    cost = cost.mean()
    # convert to bits vs nats
    cost = cost * tensor.cast(1.44269504089, theano.config.floatX)

    params = param_search(cost, lambda x: hasattr(x, "param"))
    print_param_info(params)

    grads = tensor.grad(cost, params)
    grads = [tensor.clip(g, -1., 1.) for g in grads]
    learning_rate = 1E-3
    opt = adam(params, learning_rate)
    updates = opt.updates(params, grads)

    if args.cont is not None:
        print("Continuing training from saved model")
        continue_path = args.cont
        if not os.path.exists(continue_path):
            raise ValueError("Continue model %s, path not "
                             "found" % continue_path)
        saved_checkpoint = load_checkpoint(continue_path)
        checkpoint_dict = saved_checkpoint
        train_function = checkpoint_dict["train_function"]
        cost_function = checkpoint_dict["cost_function"]
        predict_function = checkpoint_dict["predict_function"]
        sample_function = checkpoint_dict["sample_function"]
        """
        trained_weights = get_values_from_function(
            saved_checkpoint["train_function"])
        set_shared_variables_in_function(train_function, trained_weights)
        """
    else:
        train_function = theano.function([X_sym, X_mask_sym,
                                          init_h1_i, init_h2_i, init_h3_i],
                                         [cost, h1, h2, h3],
                                         updates=updates,
                                         on_unused_input="warn")
        cost_function = theano.function([X_sym, X_mask_sym,
                                         init_h1_i, init_h2_i, init_h3_i],
                                         [cost, h1, h2, h3],
                                         on_unused_input="warn")
        predict_function = theano.function([inpt,
                                          init_h1_i, init_h2_i, init_h3_i],
                                         [out, h1, h2, h3],
                                        on_unused_input="warn")
        sample_function = theano.function([inpt,
                                           init_h1_i, init_h2_i, init_h3_i],
                                          [out, h1, h2, h3],
                                          on_unused_input="warn")
        print("Beginning training loop")
        checkpoint_dict = {}
        checkpoint_dict["train_function"] = train_function
        checkpoint_dict["cost_function"] = cost_function
        checkpoint_dict["predict_function"] = predict_function
        checkpoint_dict["sample_function"] = sample_function


    def _loop(function, itr):
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]
        X_mb, X_mb_mask = next(itr)
        # sanity check that masking code is OK
        assert X_mb_mask.min() > 1E-6
        n_cuts = len(X_mb) // cut_len + 1
        partial_costs = []
        for n in range(n_cuts):
            if n % 100 == 0:
                print("step %i" % n, end="")
            else:
                print(".", end="")
            start = n * cut_len
            stop = (n + 1) * cut_len
            if len(X_mb[start:stop]) < cut_len:
                # skip end edge case
                break
            rval = function(X_mb[start:stop],
                            X_mb_mask[start:stop],
                            prev_h1, prev_h2, prev_h3)
            current_cost = rval[0]
            prev_h1, prev_h2, prev_h3 = rval[1:4]
            prev_h1 = prev_h1[-1]
            prev_h2 = prev_h2[-1]
            prev_h3 = prev_h3[-1]
        partial_costs.append(current_cost)
        print("")
        return partial_costs

i = Igor(_loop, train_function, train_itr, cost_function, valid_itr,
         n_epochs=n_epochs, checkpoint_dict=checkpoint_dict,
         checkpoint_every_n_updates=checkpoint_every_n_updates,
         checkpoint_every_n_seconds=checkpoint_every_n_seconds,
         checkpoint_every_n_epochs=checkpoint_every_n_epochs,
         skip_minimums=True)
#i.refresh(_loop, train_function, train_itr, cost_function, valid_itr,
#          n_epochs, checkpoint_dict)
i.run()
