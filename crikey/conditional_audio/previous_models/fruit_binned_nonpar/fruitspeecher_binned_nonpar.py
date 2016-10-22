import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import load_checkpoint, theano_one_hot
from kdllib import fetch_fruitspeech_spectrogram_nonpar, list_iterator
from kdllib import np_zeros, GRU, GRUFork, dense_to_one_hot
from kdllib import make_weights, make_biases, relu, run_loop
from kdllib import as_shared, adam, gradient_clipping
from kdllib import get_values_from_function, set_shared_variables_in_function
from kdllib import soundsc, categorical_crossentropy
from kdllib import sample_softmax, softmax



if __name__ == "__main__":
    import argparse

    speech = fetch_fruitspeech_spectrogram_nonpar()
    X1 = speech["data1"]
    X1_size = speech["data1_size"]
    X2 = speech["data2"]
    X2_size = speech["data2_size"]
    y = speech["target"]
    vocabulary = speech["vocabulary"]
    vocabulary_size = speech["vocabulary_size"]
    reconstruct = speech["reconstruct"]
    fs = speech["sample_rate"]
    X1 = np.array([x.astype(theano.config.floatX) for x in X1])
    X2 = np.array([x.astype(theano.config.floatX) for x in X2])
    X = np.array([np.hstack((x1[:, None], x2[:, None]))
                  for x1, x2 in zip(X1, X2)])
    y = np.array([yy.astype(theano.config.floatX) for yy in y])

    minibatch_size = 20
    n_epochs = 1000  # Used way at the bottom in the training loop!
    checkpoint_every_n = 100
    # Was 300
    cut_len = 21  # Used way at the bottom in the training loop!
    random_state = np.random.RandomState(1999)

    train_itr = list_iterator([X, y], minibatch_size, axis=1,
                              stop_index=80, make_mask=True)
    valid_itr = list_iterator([X, y], minibatch_size, axis=1,
                              start_index=80, make_mask=True)
    X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr)
    train_itr.reset()


    n_hid = 256
    att_size = 10
    n_proj = 1024
    n_softmax1 = X1_size
    n_softmax2 = X2_size
    input_dim = (n_softmax1 + n_softmax2)
    n_feats = X_mb.shape[-1]
    n_chars = vocabulary_size
    # n_components = 3
    # n_density = 2 * n_out * n_components + n_components

    desc = "Speech generation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', '--sample',
                        help='Sample from a checkpoint file',
                        default=None,
                        required=False)
    parser.add_argument('-p', '--plot',
                        help='Plot training curves from a checkpoint file',
                        default=None,
                        required=False)
    parser.add_argument('-w', '--write',
                        help='The string to write out (default first minibatch)',
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
    if args.plot is not None or args.sample is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if args.sample is not None:
            checkpoint_file = args.sample
        else:
            checkpoint_file = args.plot
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

        X_mb, X_mb_mask, c_mb, c_mb_mask = next(valid_itr)
        valid_itr.reset()
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]
        prev_kappa = np_zeros((minibatch_size, att_size))
        prev_w = np_zeros((minibatch_size, n_chars))
        if args.sample is not None:
            predict_function = checkpoint_dict["predict_function"]
            attention_function = checkpoint_dict["attention_function"]
            sample_function = checkpoint_dict["sample_function"]
            if args.write is not None:
                sample_string = args.write
                print("Sampling using sample string %s" % sample_string)
                oh = dense_to_one_hot(
                    np.array([vocabulary[c] for c in sample_string]),
                    vocabulary_size)
                c_mb = np.zeros(
                    (len(oh), minibatch_size, oh.shape[-1])).astype(c_mb.dtype)
                c_mb[:len(oh), :, :] = oh[:, None, :]
                c_mb = c_mb[:len(oh)]
                c_mb_mask = np.ones_like(c_mb[:, :, 0])

            if args.sample_length is None:
                # Automatic sampling stop as described in Graves' paper
                # Assume an average of 30 timesteps per char
                n_steps = 30 * c_mb.shape[0]
                step_inc = n_steps
                max_steps = 25000
                max_steps_buf = max_steps + n_steps
                completed = [np.zeros((max_steps_buf, X_mb.shape[-1]))
                             for i in range(c_mb.shape[1])]
                max_indices = [None] * c_mb.shape[1]
                completed_indices = set()
                # hardcoded upper limit
                while n_steps < max_steps:
                    rvals = sample_function(c_mb, c_mb_mask, prev_h1, prev_h2,
                                            prev_h3, prev_kappa, prev_w,
                                            n_steps)
                    sampled, h1_s, h2_s, h3_s, k_s, w_s, stop_s, stop_h = rvals
                    for i in range(c_mb.shape[1]):
                        max_ind = None
                        for j in range(len(stop_s)):
                            if np.all(stop_h[j, i] > stop_s[j, i]):
                                max_ind = j

                        if max_ind is not None:
                            completed_indices = completed_indices | set([i])
                            completed[i][:max_ind] = sampled[:max_ind, i]
                            max_indices[i] = max_ind
                    # if most samples meet the criteria call it good
                    if len(completed_indices) >= .8 * c_mb.shape[1]:
                        break
                    n_steps += step_inc
                print("Completed auto sampling after %i steps" % n_steps)
                # cut out garbage
                completed = [completed[i] for i in completed_indices]
                cond = c_mb[:, np.array(list(completed_indices))]
            else:
                fixed_steps = args.sample_length
                rvals = sample_function(c_mb, c_mb_mask, prev_h1, prev_h2,
                                        prev_h3, prev_kappa, prev_w,
                                        fixed_steps)
                sampled, h1_s, h2_s, h3_s, k_s, w_s, stop_s, stop_h = rvals
                completed = [sampled[:, i]
                             for i in range(sampled.shape[1])]
                cond = c_mb
                print("Completed sampling after %i steps" % fixed_steps)
            rlookup = {v: k for k, v in vocabulary.items()}
            for i in range(len(completed)):
                ex = completed[i]
                ex_str = "".join([rlookup[c]
                                  for c in np.argmax(cond[:, i], axis=1)])
                s = "gen_%s_%i.wav" % (ex_str, i)
                ii = reconstruct(ex[:, 0], ex[:, 1])
                wavfile.write(s, fs, soundsc(ii))
                #it = reconstruct(X[0])
                #wavfile.write("orig.wav", fs, soundsc(it))
                # plot_lines_iamondb_example(ex, title=ex_str, save_name=s)
        valid_itr.reset()
        print("Sampling complete, exiting...")
        sys.exit()
    else:
        print("No plotting arguments, starting training mode!")

    X_sym = tensor.tensor3("X_sym")
    X_sym.tag.test_value = X_mb
    X_mask_sym = tensor.matrix("X_mask_sym")
    X_mask_sym.tag.test_value = X_mb_mask
    c_sym = tensor.tensor3("c_sym")
    c_sym.tag.test_value = c_mb
    c_mask_sym = tensor.matrix("c_mask_sym")
    c_mask_sym.tag.test_value = c_mb_mask

    init_h1 = tensor.matrix("init_h1")
    init_h1.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h2 = tensor.matrix("init_h2")
    init_h2.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h3 = tensor.matrix("init_h3")
    init_h3.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_kappa = tensor.matrix("init_kappa")
    init_kappa.tag.test_value = np_zeros((minibatch_size, att_size))

    init_w = tensor.matrix("init_w")
    init_w.tag.test_value = np_zeros((minibatch_size, n_chars))

    params = []
    biases = []

    cell1 = GRU(input_dim, n_hid, random_state)
    cell2 = GRU(n_hid, n_hid, random_state)
    cell3 = GRU(n_hid, n_hid, random_state)

    params += cell1.get_params()
    params += cell2.get_params()
    params += cell3.get_params()

    # Use GRU classes only to fork 1 inp to 2 inp:gate pairs
    inp_proj, = make_weights(input_dim, [n_hid], random_state)
    inp_b, = make_biases([n_hid])

    params += [inp_proj, inp_b]
    biases += [inp_b]

    inp_to_h1 = GRUFork(n_hid, n_hid, random_state)
    inp_to_h2 = GRUFork(n_hid, n_hid, random_state)
    inp_to_h3 = GRUFork(n_hid, n_hid, random_state)
    att_to_h1 = GRUFork(n_chars, n_hid, random_state)
    att_to_h2 = GRUFork(n_chars, n_hid, random_state)
    att_to_h3 = GRUFork(n_chars, n_hid, random_state)
    h1_to_h2 = GRUFork(n_hid, n_hid, random_state)
    h1_to_h3 = GRUFork(n_hid, n_hid, random_state)
    h2_to_h3 = GRUFork(n_hid, n_hid, random_state)

    params += inp_to_h1.get_params()
    params += inp_to_h2.get_params()
    params += inp_to_h3.get_params()
    params += att_to_h1.get_params()
    params += att_to_h2.get_params()
    params += att_to_h3.get_params()
    params += h1_to_h2.get_params()
    params += h1_to_h3.get_params()
    params += h2_to_h3.get_params()

    biases += inp_to_h1.get_biases()
    biases += inp_to_h2.get_biases()
    biases += inp_to_h3.get_biases()
    biases += att_to_h1.get_biases()
    biases += att_to_h2.get_biases()
    biases += att_to_h3.get_biases()
    biases += h1_to_h2.get_biases()
    biases += h1_to_h3.get_biases()
    biases += h2_to_h3.get_biases()

    h1_to_att_a, h1_to_att_b, h1_to_att_k = make_weights(n_hid, 3 * [att_size],
                                                         random_state)
    h1_to_outs, = make_weights(n_hid, [n_proj], random_state)
    h2_to_outs, = make_weights(n_hid, [n_proj], random_state)
    h3_to_outs, = make_weights(n_hid, [n_proj], random_state)

    params += [h1_to_att_a, h1_to_att_b, h1_to_att_k]
    params += [h1_to_outs, h2_to_outs, h3_to_outs]

    # Not used
    l1_proj, l2_proj = make_weights(n_proj, [n_proj, n_proj], random_state,
                                    init="fan")
    l1_b, l2_b = make_biases([n_proj, n_proj])
    #params += [l1_proj, l1_b, l2_proj, l2_b]

    softmax1_proj, = make_weights(n_proj, [n_softmax1], random_state)
    softmax1_b, = make_biases([n_softmax1])
    softmax2_proj, = make_weights(n_proj, [n_softmax2], random_state)
    softmax2_b, = make_biases([n_softmax2])

    params += [softmax1_proj, softmax1_b, softmax2_proj, softmax2_b]
    biases += [softmax1_b, softmax2_b]

    inpt = X_sym[:-1]
    target = X_sym[1:]
    mask = X_mask_sym[1:]
    context = c_sym * c_mask_sym.dimshuffle(0, 1, 'x')

    pt1 = theano_one_hot(inpt[:, :, 0], n_classes=n_softmax1)
    pt2 = theano_one_hot(inpt[:, :, 1], n_classes=n_softmax2)
    inpt = tensor.concatenate((pt1, pt2), axis=-1)
    inpt_reduced = inpt.dot(inp_proj) + inp_b
    inp_h1, inpgate_h1 = inp_to_h1.proj(inpt_reduced)
    inp_h2, inpgate_h2 = inp_to_h2.proj(inpt_reduced)
    inp_h3, inpgate_h3 = inp_to_h3.proj(inpt_reduced)

    u = tensor.arange(c_sym.shape[0]).dimshuffle('x', 'x', 0)
    u = tensor.cast(u, theano.config.floatX)

    def calc_phi(k_t, a_t, b_t, u_c):
        a_t = a_t.dimshuffle(0, 1, 'x')
        b_t = b_t.dimshuffle(0, 1, 'x')
        ss1 = (k_t.dimshuffle(0, 1, 'x') - u_c) ** 2
        ss2 = -b_t * ss1
        ss3 = a_t * tensor.exp(ss2)
        ss4 = ss3.sum(axis=1)
        return ss4

    def step(xinp_h1_t, xgate_h1_t,
             xinp_h2_t, xgate_h2_t,
             xinp_h3_t, xgate_h3_t,
             h1_tm1, h2_tm1, h3_tm1,
             k_tm1, w_tm1, ctx):

        attinp_h1, attgate_h1 = att_to_h1.proj(w_tm1)

        h1_t = cell1.step(xinp_h1_t + attinp_h1, xgate_h1_t + attgate_h1,
                          h1_tm1)
        h1inp_h2, h1gate_h2 = h1_to_h2.proj(h1_t)
        h1inp_h3, h1gate_h3 = h1_to_h3.proj(h1_t)

        a_t = h1_t.dot(h1_to_att_a)
        b_t = h1_t.dot(h1_to_att_b)
        k_t = h1_t.dot(h1_to_att_k)

        a_t = tensor.exp(a_t)
        b_t = tensor.exp(b_t)
        k_t = k_tm1 + tensor.exp(k_t)

        ss4 = calc_phi(k_t, a_t, b_t, u)
        ss5 = ss4.dimshuffle(0, 1, 'x')
        ss6 = ss5 * ctx.dimshuffle(1, 0, 2)
        w_t = ss6.sum(axis=1)

        attinp_h2, attgate_h2 = att_to_h2.proj(w_t)
        attinp_h3, attgate_h3 = att_to_h3.proj(w_t)

        h2_t = cell2.step(xinp_h2_t + h1inp_h2 + attinp_h2,
                          xgate_h2_t + h1gate_h2 + attgate_h2, h2_tm1)

        h2inp_h3, h2gate_h3 = h2_to_h3.proj(h2_t)

        h3_t = cell3.step(xinp_h3_t + h1inp_h3 + h2inp_h3 + attinp_h3,
                          xgate_h3_t + h1gate_h3 + h2gate_h3 + attgate_h3,
                          h3_tm1)
        return h1_t, h2_t, h3_t, k_t, w_t

    init_x = as_shared(np_zeros((minibatch_size, n_feats)))
    srng = RandomStreams(1999)

    # Used to calculate stopping heuristic from sections 5.3
    u_max = 0. * tensor.arange(c_sym.shape[0]) + c_sym.shape[0]
    u_max = u_max.dimshuffle('x', 'x', 0)
    u_max = tensor.cast(u_max, theano.config.floatX)

    def sample_step(x_tm1, h1_tm1, h2_tm1, h3_tm1, k_tm1, w_tm1, ctx):
        theano.printing.Print("x_tm1.shape")(x_tm1.shape)
        pt1 = theano_one_hot(x_tm1[:, 0],
                             n_classes=n_softmax1)
        theano.printing.Print("pt1.shape")(pt1.shape)
        pt2 = theano_one_hot(x_tm1[:, 1],
                             n_classes=n_softmax2)
        theano.printing.Print("pt2.shape")(pt2.shape)
        x_tm1 = tensor.concatenate((pt1, pt2), axis=-1)
        theano.printing.Print("x_tm1.shape")(x_tm1.shape)
        x_tm1_reduced = x_tm1.dot(inp_proj) + inp_b
        theano.printing.Print("x_tm1_reduced.shape")(x_tm1_reduced.shape)
        xinp_h1_t, xgate_h1_t = inp_to_h1.proj(x_tm1_reduced)
        xinp_h2_t, xgate_h2_t = inp_to_h2.proj(x_tm1_reduced)
        xinp_h3_t, xgate_h3_t = inp_to_h3.proj(x_tm1_reduced)

        attinp_h1, attgate_h1 = att_to_h1.proj(w_tm1)

        h1_t = cell1.step(xinp_h1_t + attinp_h1, xgate_h1_t + attgate_h1,
                          h1_tm1)
        h1inp_h2, h1gate_h2 = h1_to_h2.proj(h1_t)
        h1inp_h3, h1gate_h3 = h1_to_h3.proj(h1_t)

        a_t = h1_t.dot(h1_to_att_a)
        b_t = h1_t.dot(h1_to_att_b)
        k_t = h1_t.dot(h1_to_att_k)

        a_t = tensor.exp(a_t)
        b_t = tensor.exp(b_t)
        k_t = k_tm1 + tensor.exp(k_t)

        ss_t = calc_phi(k_t, a_t, b_t, u)
        # calculate and return stopping criteria
        sh_t = calc_phi(k_t, a_t, b_t, u_max)
        ss5 = ss_t.dimshuffle(0, 1, 'x')
        ss6 = ss5 * ctx.dimshuffle(1, 0, 2)
        w_t = ss6.sum(axis=1)

        attinp_h2, attgate_h2 = att_to_h2.proj(w_t)
        attinp_h3, attgate_h3 = att_to_h3.proj(w_t)

        h2_t = cell2.step(xinp_h2_t + h1inp_h2 + attinp_h2,
                          xgate_h2_t + h1gate_h2 + attgate_h2, h2_tm1)

        h2inp_h3, h2gate_h3 = h2_to_h3.proj(h2_t)

        h3_t = cell3.step(xinp_h3_t + h1inp_h3 + h2inp_h3 + attinp_h3,
                          xgate_h3_t + h1gate_h3 + h2gate_h3 + attgate_h3,
                          h3_tm1)
        out_t = h1_t.dot(h1_to_outs) + h2_t.dot(h2_to_outs) + h3_t.dot(
            h3_to_outs)
        #l1_t = relu(out_t.dot(l1_proj) + l1_b)
        #l2_t = relu(l1_t.dot(l2_proj) + l2_b)
        pred1_t = softmax(out_t.dot(softmax1_proj) + softmax1_b)
        pred2_t = softmax(out_t.dot(softmax2_proj) + softmax2_b)

        s1_t = sample_softmax(pred1_t, srng)
        theano.printing.Print("s1_t.shape")(s1_t.shape)
        s2_t = sample_softmax(pred2_t, srng)
        theano.printing.Print("s2_t.shape")(s2_t.shape)
        s1_t = s1_t.dimshuffle(0, 'x')
        theano.printing.Print("s1_t.shape")(s1_t.shape)
        s2_t = s2_t.dimshuffle(0, 'x')
        theano.printing.Print("s2_t.shape")(s2_t.shape)
        x_t = tensor.concatenate((s1_t, s2_t), axis=1)
        theano.printing.Print("x_t.shape")(x_t.shape)
        return x_t, h1_t, h2_t, h3_t, k_t, w_t, ss_t, sh_t


    n_steps_sym = tensor.iscalar()
    n_steps_sym.tag.test_value = 10
    # Can't do test values with sampling?
    (sampled, h1_s, h2_s, h3_s, k_s, w_s, stop_s, stop_h), supdates = theano.scan(
        fn=sample_step,
        n_steps=n_steps_sym,
        sequences=[],
        outputs_info=[init_x, init_h1, init_h2, init_h3,
                      init_kappa, init_w, None, None],
        non_sequences=[context])
    theano.printing.Print("sampled.shape")(sampled.shape)

    """
    # Testing step function
    r = step(inp_h1[0], inpgate_h1[0], inp_h2[0], inpgate_h2[0],
             inp_h3[0], inpgate_h3[0],
             init_h1, init_h2, init_h3, init_kappa, init_w, context)

    r = step(inp_h1[1], inpgate_h1[1], inp_h2[1], inpgate_h2[1],
             inp_h3[1], inpgate_h3[1],
             r[0], r[1], r[2], r[3], r[4], context)
    """
    (h1, h2, h3, kappa, w), updates = theano.scan(
        fn=step,
        sequences=[inp_h1, inpgate_h1,
                   inp_h2, inpgate_h2,
                   inp_h3, inpgate_h3],
        outputs_info=[init_h1, init_h2, init_h3, init_kappa, init_w],
        non_sequences=[context])

    outs = h1.dot(h1_to_outs) + h2.dot(h2_to_outs) + h3.dot(h3_to_outs)
    theano.printing.Print("outs.shape")(outs.shape)
    outs_shape = outs.shape

    #l1 = relu(outs.dot(l1_proj) + l1_b)
    #l2 = relu(l1.dot(l2_proj) + l2_b)
    #theano.printing.Print("l1.shape")(l1.shape)
    #theano.printing.Print("l2.shape")(l2.shape)

    pred1 = softmax(outs.dot(softmax1_proj) + softmax1_b)
    pred2 = softmax(outs.dot(softmax2_proj) + softmax2_b)
    theano.printing.Print("pred1.shape")(pred1.shape)
    theano.printing.Print("pred2.shape")(pred2.shape)


    # Make one hot targets
    theano.printing.Print("target.shape")(target.shape)
    target1 = target[:, :, 0]
    target2 = target[:, :, 1]
    theano.printing.Print("target1.shape")(target1.shape)
    theano.printing.Print("target2.shape")(target2.shape)

    shp = target1.shape
    target1 = target1.ravel()
    target1 = theano_one_hot(target1, n_classes=n_softmax1)
    target1 = target1.reshape((shp[0], shp[1], n_softmax1))
    theano.printing.Print("target1.shape")(target1.shape)

    shp = target2.shape
    target2 = target2.ravel()
    target2 = theano_one_hot(target2, n_classes=n_softmax2)
    target2 = target2.reshape((shp[0], shp[1], n_softmax2))
    theano.printing.Print("target2.shape")(target2.shape)

    cost1 = categorical_crossentropy(pred1, target1)
    cost2 = categorical_crossentropy(pred2, target2)
    theano.printing.Print("cost1.shape")(cost1.shape)
    theano.printing.Print("cost2.shape")(cost2.shape)

    # Used to upweight cost1
    ratio1 = n_softmax2 / float(n_softmax1 + n_softmax2)
    # Used to downweight cost2
    ratio2 = n_softmax1 / float(n_softmax1 + n_softmax2)

    cost = cost1 * ratio1 + cost2 * ratio2
    cost = cost * mask
    cost = cost.sum() / (cut_len * minibatch_size)

    l2_penalty = 0
    for p in list(set(params) - set(biases)):
        l2_penalty += (p ** 2).sum()

    cost = cost + 1E-3 * l2_penalty
    grads = tensor.grad(cost, params)
    grads = gradient_clipping(grads, 10.)

    learning_rate = 1E-4

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
        attention_function = checkpoint_dict["attention_function"]
        sample_function = checkpoint_dict["sample_function"]
        """
        trained_weights = get_values_from_function(
            saved_checkpoint["train_function"])
        set_shared_variables_in_function(train_function, trained_weights)
        """
    else:
        train_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                          init_h1, init_h2, init_h3, init_kappa,
                                          init_w],
                                         [cost, h1, h2, h3, kappa, w],
                                         updates=updates)
        cost_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                         init_h1, init_h2, init_h3, init_kappa,
                                         init_w],
                                        [cost, h1, h2, h3, kappa, w])
        predict_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                            init_h1, init_h2, init_h3, init_kappa,
                                            init_w],
                                           [outs],
                                           on_unused_input='warn')
        attention_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                              init_h1, init_h2, init_h3, init_kappa,
                                              init_w],
                                             [kappa, w], on_unused_input='warn')
        sample_function = theano.function([c_sym, c_mask_sym, init_h1, init_h2,
                                           init_h3, init_kappa, init_w,
                                           n_steps_sym],
                                          [sampled, h1_s, h2_s, h3_s, k_s, w_s,
                                           stop_s, stop_h],
                                          updates=supdates)
        print("Beginning training loop")
        checkpoint_dict = {}
        checkpoint_dict["train_function"] = train_function
        checkpoint_dict["cost_function"] = cost_function
        checkpoint_dict["predict_function"] = predict_function
        checkpoint_dict["attention_function"] = attention_function
        checkpoint_dict["sample_function"] = sample_function


    def _loop(function, itr):
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]
        prev_kappa = np_zeros((minibatch_size, att_size))
        prev_w = np_zeros((minibatch_size, n_chars))
        X_mb, X_mb_mask, c_mb, c_mb_mask = next(itr)
        n_cuts = len(X_mb) // cut_len + 1
        partial_costs = []
        for n in range(n_cuts):
            start = n * cut_len
            stop = (n + 1) * cut_len
            if len(X_mb[start:stop]) <= 1:
                break
            """
            # Extended masking breaks learning? Wut
            if len(X_mb[start:stop]) <= cut_len:
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
            """
            rval = function(X_mb[start:stop],
                            X_mb_mask[start:stop],
                            c_mb, c_mb_mask,
                            prev_h1, prev_h2, prev_h3, prev_kappa, prev_w)
            current_cost = rval[0]
            prev_h1, prev_h2, prev_h3 = rval[1:4]
            prev_h1 = prev_h1[-1]
            prev_h2 = prev_h2[-1]
            prev_h3 = prev_h3[-1]
            prev_kappa = rval[4][-1]
            prev_w = rval[5][-1]
        partial_costs.append(current_cost)
        return partial_costs

run_loop(_loop, train_function, train_itr, cost_function, valid_itr,
         n_epochs=n_epochs, checkpoint_dict=checkpoint_dict,
         checkpoint_every_n=checkpoint_every_n)
