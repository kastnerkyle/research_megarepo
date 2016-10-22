import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import overlap
from kdllib import load_checkpoint, theano_one_hot, concatenate
from kdllib import fetch_fruitspeech, list_iterator
from kdllib import np_zeros, GRU, GRUFork, dense_to_one_hot
from kdllib import make_weights, make_biases, relu, run_loop
from kdllib import as_shared, adam, gradient_clipping
from kdllib import get_values_from_function, set_shared_variables_in_function
from kdllib import soundsc, categorical_crossentropy
from kdllib import relu, softmax, sample_softmax



if __name__ == "__main__":
    import argparse

    speech = fetch_fruitspeech()
    X = speech["data"]
    fs = speech["sample_rate"]
    reconstruct = speech["reconstruct"]
    X = np.array([x.astype(theano.config.floatX) for x in X])
    # reslice to 2D in sets of 128, 4
    def _s(x):
        xi = x[:len(x) - len(x) % 4]
        # chunk it
        xi = xi.reshape((-1, 4))
        #xi = overlap(xi, 4, 1)
        return xi

    X_windows = []
    for x in X:
        xw = _s(x)
        X_windows.append(xw)
    X = X_windows

    y = [Xi[1:, 0][:, None] for Xi in X]
    X = [Xi[:-1] for Xi in X]

    minibatch_size = 1
    n_epochs = 700  # Used way at the bottom in the training loop!
    checkpoint_every_n = 10
    random_state = np.random.RandomState(1999)

    train_itr = list_iterator([X, y], minibatch_size, axis=1,
                              stop_index=4, randomize=True, make_mask=True)
    valid_itr = list_iterator([X, y], minibatch_size, axis=1,
                              stop_index=4,
                              randomize=True, make_mask=True)
    X_mb, X_mb_mask, y_mb, y_mb_mask = next(train_itr)
    train_itr.reset()
    n_hid = 512
    n_frame = 4
    n_proj = n_frame * n_hid
    n_bins = 256
    input_dim = X_mb.shape[-1]
    n_inpt_mlp = n_bins * n_frame + n_hid
    n_hid_mlp = 1024
    n_pred_proj = n_bins
    cut_len = 60

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
        train_costs = checkpoint_dict["train_costs"]
        valid_costs = checkpoint_dict["valid_costs"]
        plt.plot(train_costs)
        plt.plot(valid_costs)
        plt.savefig("costs.png")

        X_mb, X_mb_mask, y_mb, y_mb_mask = next(train_itr)
        train_itr.reset()
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]

        predict_function = checkpoint_dict["predict_function"]
        sample_function = checkpoint_dict["sample_function"]

        if args.sample_length is None:
            raise ValueError("NYI - use -sl or --sample_length ")
        else:
            fixed_steps = args.sample_length
            completed = []
            init_x = np.zeros_like(X_mb[0]) + int(n_bins // 2)
            for i in range(fixed_steps):
                if i % 100 == 0:
                    print("Sampling step %i" % i)
                # remove second init_x later
                rvals = sample_function(init_x, prev_h1, prev_h2,
                                        prev_h3)
                sampled, h1_s, h2_s, h3_s = rvals
                # remove cast later
                sampled = sampled.astype("float32")
                completed.append(sampled)
                # cheating sampling...
                #init_x = X_mb[i]
                init_x = sampled
                prev_h1 = h1_s
                prev_h2 = h2_s
                prev_h3 = h3_s
            print("Completed sampling after %i steps" % fixed_steps)
            completed = np.array(completed).transpose(1, 0, 2)
            for i in range(len(completed)):
                ex = completed[i].ravel()
                s = "gen_%i.wav" % (i)
                ii = reconstruct(ex.astype("int32"))
                wavfile.write(s, fs, soundsc(ii))
        print("Sampling complete, exiting...")
        sys.exit()
    else:
        print("No plotting arguments, starting training mode!")

    X_sym = tensor.tensor3("X_sym")
    X_sym.tag.test_value = X_mb[:cut_len]
    X_mask_sym = tensor.matrix("X_mask_sym")
    X_mask_sym.tag.test_value = X_mb_mask[:cut_len]
    y_sym = tensor.tensor3("y_sym")
    y_sym.tag.test_value = y_mb[:cut_len]
    y_mask_sym = tensor.matrix("y_mask_sym")
    y_mask_sym.tag.test_value = y_mb_mask[:cut_len]

    init_h1 = tensor.matrix("init_h1")
    init_h1.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h2 = tensor.matrix("init_h2")
    init_h2.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h3 = tensor.matrix("init_h3")
    init_h3.tag.test_value = np_zeros((minibatch_size, n_hid))

    params = []
    biases = []

    cell1 = GRU(input_dim, n_hid, random_state)
    cell2 = GRU(n_hid, n_hid, random_state)
    cell3 = GRU(n_hid, n_hid, random_state)

    params += cell1.get_params()
    params += cell2.get_params()
    params += cell3.get_params()

    inp_to_h1 = GRUFork(input_dim, n_hid, random_state)
    inp_to_h2 = GRUFork(input_dim, n_hid, random_state)
    inp_to_h3 = GRUFork(input_dim, n_hid, random_state)
    h1_to_h2 = GRUFork(n_hid, n_hid, random_state)
    h1_to_h3 = GRUFork(n_hid, n_hid, random_state)
    h2_to_h3 = GRUFork(n_hid, n_hid, random_state)

    params += inp_to_h1.get_params()
    params += inp_to_h2.get_params()
    params += inp_to_h3.get_params()
    params += h1_to_h2.get_params()
    params += h1_to_h3.get_params()
    params += h2_to_h3.get_params()

    biases += inp_to_h1.get_biases()
    biases += inp_to_h2.get_biases()
    biases += inp_to_h3.get_biases()
    biases += h1_to_h2.get_biases()
    biases += h1_to_h3.get_biases()
    biases += h2_to_h3.get_biases()

    h1_to_outs, = make_weights(n_hid, [n_proj], random_state)
    h2_to_outs, = make_weights(n_hid, [n_proj], random_state)
    h3_to_outs, = make_weights(n_hid, [n_proj], random_state)
    b_to_outs, = make_biases([n_proj])

    params += [h1_to_outs, h2_to_outs, h3_to_outs]
    biases += [b_to_outs]

    mlp1_w, = make_weights(n_inpt_mlp, [n_hid_mlp], random_state)
    mlp2_w, mlp3_w = make_weights(n_hid_mlp, 2 * [n_hid_mlp], random_state)
    pred_w, = make_weights(n_hid_mlp, [n_bins], random_state)

    mlp1_b, mlp2_b, mlp3_b = make_biases(3 * [n_hid_mlp])
    pred_b, = make_biases([n_bins])

    params += [mlp1_w, mlp1_b]
    params += [mlp2_w, mlp2_b]
    params += [mlp3_w, mlp3_b]
    params += [pred_w, pred_b]
    biases += [mlp1_b, mlp2_b, mlp3_b, pred_b]

    # Done initializing

    inpt = X_sym[:-1]
    target = X_sym[1:]
    #target = y_sym
    mask = X_mask_sym[:-1]

    inp_h1, inpgate_h1 = inp_to_h1.proj(inpt)
    inp_h2, inpgate_h2 = inp_to_h2.proj(inpt)
    inp_h3, inpgate_h3 = inp_to_h3.proj(inpt)

    def step(xinp_h1_t, xgate_h1_t,
             xinp_h2_t, xgate_h2_t,
             xinp_h3_t, xgate_h3_t,
             h1_tm1, h2_tm1, h3_tm1):
        h1_t = cell1.step(xinp_h1_t, xgate_h1_t,
                          h1_tm1)
        h1inp_h2, h1gate_h2 = h1_to_h2.proj(h1_t)
        h1inp_h3, h1gate_h3 = h1_to_h3.proj(h1_t)

        h2_t = cell2.step(xinp_h2_t + h1inp_h2,
                          xgate_h2_t + h1gate_h2, h2_tm1)

        h2inp_h3, h2gate_h3 = h2_to_h3.proj(h2_t)

        h3_t = cell3.step(xinp_h3_t + h1inp_h3 + h2inp_h3,
                          xgate_h3_t + h1gate_h3 + h2gate_h3,
                          h3_tm1)
        return h1_t, h2_t, h3_t


    init_x = tensor.fmatrix()
    init_x.tag.test_value = np_zeros((minibatch_size, input_dim)).astype(theano.config.floatX)
    srng = RandomStreams(1999)

    def sample_step(x_tm1, h1_tm1, h2_tm1, h3_tm1):
        xinp_h1_t, xgate_h1_t = inp_to_h1.proj(x_tm1)
        xinp_h2_t, xgate_h2_t = inp_to_h2.proj(x_tm1)
        xinp_h3_t, xgate_h3_t = inp_to_h3.proj(x_tm1)

        h1_t = cell1.step(xinp_h1_t, xgate_h1_t, h1_tm1)
        h1inp_h2, h1gate_h2 = h1_to_h2.proj(h1_t)
        h1inp_h3, h1gate_h3 = h1_to_h3.proj(h1_t)


        h2_t = cell2.step(xinp_h2_t + h1inp_h2,
                          xgate_h2_t + h1gate_h2, h2_tm1)

        h2inp_h3, h2gate_h3 = h2_to_h3.proj(h2_t)

        h3_t = cell3.step(xinp_h3_t + h1inp_h3 + h2inp_h3,
                          xgate_h3_t + h1gate_h3 + h2gate_h3,
                          h3_tm1)
        out_t = h1_t.dot(h1_to_outs) + h2_t.dot(h2_to_outs) + h3_t.dot(
            h3_to_outs) + b_to_outs

        theano.printing.Print("x_tm1.shape")(x_tm1.shape)

        theano.printing.Print("out_t.shape")(out_t.shape)
        inpt_oh = theano_one_hot(x_tm1, n_classes=n_bins)
        theano.printing.Print("inpt_oh.shape")(inpt_oh.shape)
        shp = inpt_oh.shape
        prev_t = inpt_oh
        for i in range(n_frame):
            partial_out_t = out_t[:, i * n_hid: (i + 1) * n_hid]
            theano.printing.Print("partial_out_t.shape")(partial_out_t.shape)
            theano.printing.Print("prev_t.shape")(prev_t.shape)
            shp = prev_t.shape
            prev_ti = prev_t[:, i:].reshape((shp[0], -1))
            theano.printing.Print("prev_ti.shape")(prev_ti.shape)
            features_t = tensor.concatenate((partial_out_t, prev_ti),
                         axis=1)
            theano.printing.Print("features_t.shape")(features_t.shape)
            mlp1_t = relu(features_t.dot(mlp1_w) + mlp1_b)
            mlp2_t = relu(mlp1_t.dot(mlp2_w) + mlp2_b)
            mlp3_t = relu(mlp2_t.dot(mlp3_w) + mlp3_b)
            pred_t = softmax(mlp3_t.dot(pred_w) + pred_b)
            theano.printing.Print("pred_t.shape")(pred_t.shape)
            samp_t = sample_softmax(pred_t, srng)
            theano.printing.Print("samp_t.shape")(samp_t.shape)
            samp_t_oh = theano_one_hot(samp_t, n_classes=n_bins)
            samp_t_oh = samp_t_oh.dimshuffle(0, 'x', 1)
            theano.printing.Print("samp_t_oh.shape")(samp_t_oh.shape)
            prev_t = tensor.concatenate((prev_t, samp_t_oh), axis=1)
            theano.printing.Print("prev_t.shape")(prev_t.shape)
        pred_t = prev_t[:, n_frame:].argmax(axis=-1)
        x_t = tensor.cast(pred_t, theano.config.floatX)
        return x_t, h1_t, h2_t, h3_t

    (sampled, h1_s, h2_s, h3_s) = sample_step(
        init_x, init_h1, init_h2, init_h3)
    theano.printing.Print("sampled.shape")(sampled.shape)

    (h1, h2, h3), updates = theano.scan(
        fn=step,
        sequences=[inp_h1, inpgate_h1,
                   inp_h2, inpgate_h2,
                   inp_h3, inpgate_h3],
        outputs_info=[init_h1, init_h2, init_h3])

    outs = h1.dot(h1_to_outs) + h2.dot(h2_to_outs) + h3.dot(h3_to_outs) + b_to_outs
    outs_shape = outs.shape
    last_samp = tensor.zeros_like(inpt[0]).dimshuffle('x', 0, 1)
    next_inpt = target
    inpt_oh = theano_one_hot(inpt, n_classes=n_bins)
    next_oh = theano_one_hot(target, n_classes=n_bins)
    theano.printing.Print("inpt_oh.shape")(inpt_oh.shape)
    theano.printing.Print("next_oh.shape")(next_oh.shape)
    pred_i = []
    for i in range(n_frame):
        partial_outs = outs[:, :, i * n_hid: (i + 1) * n_hid]
        joint = tensor.concatenate((inpt_oh, next_oh), axis=2)
        sliced_context = joint[:, :, i:i + n_frame]
        theano.printing.Print("sliced_context.shape")(sliced_context.shape)
        shp = sliced_context.shape
        sliced_context = sliced_context.reshape((shp[0], shp[1], -1))
        features = tensor.concatenate((partial_outs, sliced_context), axis=2)
        theano.printing.Print("partial_outs.shape")(partial_outs.shape)
        theano.printing.Print("joint.shape")(joint.shape)
        theano.printing.Print("sliced_context.shape")(sliced_context.shape)
        theano.printing.Print("features.shape")(features.shape)
        shp = features.shape
        mlp_inpt = features.reshape((-1, shp[-1]))
        mlp1 = relu(mlp_inpt.dot(mlp1_w) + mlp1_b)
        mlp2 = relu(mlp1.dot(mlp2_w) + mlp2_b)
        mlp3 = relu(mlp2.dot(mlp3_w) + mlp3_b)
        pred = softmax(mlp3.dot(pred_w) + pred_b)
        theano.printing.Print("pred.shape")(pred.shape)
        pred = pred.reshape((shp[0], shp[1], -1))
        theano.printing.Print("pred.shape")(pred.shape)
        pred_i.append(pred.dimshuffle(0, 1, 2, 'x'))
    pred = tensor.concatenate(pred_i, axis=-1).dimshuffle(0, 1, 3, 2)
    theano.printing.Print("pred.shape")(pred.shape)
    theano.printing.Print("target.shape")(target.shape)
    target = theano_one_hot(target, n_classes=n_bins)
    theano.printing.Print("target.shape")(target.shape)
    # dimshuffle so batch is on last axis
    cost = categorical_crossentropy(pred, target, eps=1E-9)
    theano.printing.Print("cost.shape")(cost.shape)
    theano.printing.Print("mask.shape")(mask.shape)

    cost = cost * mask.dimshuffle(0, 1, 'x')
    # sum over sequence length and features, mean over minibatch
    cost = cost.dimshuffle(0, 2, 1)
    shp = cost.shape
    cost = cost.reshape((-1, shp[-1]))
    theano.printing.Print("cost.shape")(cost.shape)
    # sum over sequence mean over minibatch
    cost = cost.mean()

    l2_penalty = 0
    for p in list(set(params) - set(biases)):
        l2_penalty += (p ** 2).sum()

    cost = cost + 1E-3 * l2_penalty
    grads = tensor.grad(cost, params)
    grads = [tensor.clip(g, -1, 1) for g in grads]
    #grads = gradient_clipping(grads, 10.)

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
        train_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym,
                                          init_h1, init_h2, init_h3],
                                         [cost, h1, h2, h3],
                                         updates=updates,
                                         on_unused_input="warn")
        cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym,
                                          init_h1, init_h2, init_h3],
                                         [cost, h1, h2, h3],
                                         on_unused_input="warn")
        predict_function = theano.function([X_sym, X_mask_sym,
                                          init_h1, init_h2, init_h3],
                                         [pred, h1, h2, h3],
                                        on_unused_input="warn")
        sample_function = theano.function([init_x, init_h1, init_h2,
                                           init_h3],
                                          [sampled, h1_s, h2_s, h3_s],
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
        X_mb, X_mb_mask, y_mb, y_mb_mask = next(itr)
        n_cuts = len(X_mb) // cut_len + 1
        partial_costs = []
        for n in range(n_cuts):
            start = n * cut_len
            stop = (n + 1) * cut_len
            if len(X_mb[start:stop]) < cut_len:
                # skip end edge case
                break
                new_len = cut_len - len(X_mb) % cut_len
                zeros = np.zeros((new_len, X_mb.shape[1],
                                  X_mb.shape[2]))
                zeros = zeros.astype(X_mb.dtype)
                mask_zeros = np.zeros((new_len, X_mb_mask.shape[1]))
                mask_zeros = mask_zeros.astype(X_mb_mask.dtype)
                X_mb = np.concatenate((X_mb, zeros), axis=0)
                X_mb_mask = np.concatenate((X_mb_mask, mask_zeros), axis=0)
                assert len(X_mb[start:stop]) == cut_len
                assert len(X_mb_mask[start:stop]) == cut_len
            rval = function(X_mb[start:stop],
                            X_mb_mask[start:stop],
                            y_mb[start:stop], y_mb_mask[start:stop],
                            prev_h1, prev_h2, prev_h3)
            current_cost = rval[0]
            prev_h1, prev_h2, prev_h3 = rval[1:4]
            prev_h1 = prev_h1[-1]
            prev_h2 = prev_h2[-1]
            prev_h3 = prev_h3[-1]
        partial_costs.append(current_cost)
        return partial_costs

run_loop(_loop, train_function, train_itr, cost_function, valid_itr,
         n_epochs=n_epochs, checkpoint_dict=checkpoint_dict,
         checkpoint_every_n=checkpoint_every_n, skip_minimums=True)
