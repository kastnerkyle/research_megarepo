import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import load_checkpoint, dense_to_one_hot, plot_lines_iamondb_example
from kdllib import fetch_walla, list_iterator, np_zeros, GRU, GRUFork
from kdllib import make_weights, as_shared, adam, gradient_clipping
from kdllib import get_values_from_function, set_shared_variables_in_function
from kdllib import save_checkpoint, save_weights, sample_diagonal_gmm
from kdllib import diagonal_gmm, diagonal_phase_gmm, soundsc


if __name__ == "__main__":
    import argparse

    speech = fetch_walla()
    X = speech["data"]
    y = speech["target"]
    vocabulary = speech["vocabulary"]
    vocabulary_size = speech["vocabulary_size"]
    reconstruct = speech["reconstruct"]
    fs = speech["sample_rate"]
    X = [x.astype(theano.config.floatX) for x in X]
    y = [yy.astype(theano.config.floatX) for yy in y]
    

    minibatch_size = 20
    n_epochs = 20000  # Used way at the bottom in the training loop!
    checkpoint_every_n = 500
    # Was 300
    cut_len = 41  # Used way at the bottom in the training loop!
    random_state = np.random.RandomState(1999)

    train_itr = list_iterator([X, y], minibatch_size, axis=1, stop_index=100,
                              randomize=True, make_mask=True)
    valid_itr = list_iterator([X, y], minibatch_size, axis=1, start_index=100,
                              make_mask=True)

    X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr)
    train_itr.reset()

    input_dim = X_mb.shape[-1]
    n_hid = 400
    n_v_hid = 100
    att_size = 10
    n_components = 20
    n_out = X_mb.shape[-1]
    n_chars = vocabulary_size
    # mag and phase each n_out // 2
    # one 2 for mu, sigma , + n_components for coeff
    n_density = 2 * n_out // 2 * n_components + n_components

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
                        help='The string to use',
                        default=None,
                        required=False)
    # http://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin

    def restricted_float(x):
        x = float(x)
        if x < 0.0:
            raise argparse.ArgumentTypeError("%r not range [0.0, inf]" % (x,))
        return x
    parser.add_argument('-b', '--bias',
                        help='Bias parameter as a float',
                        type=restricted_float,
                        default=.1,
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
        train_costs = checkpoint_dict["overall_train_costs"]
        valid_costs = checkpoint_dict["overall_valid_costs"]
        plt.plot(train_costs)
        plt.plot(valid_costs)
        plt.savefig("costs.png")

        X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr)
        train_itr.reset()
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]
        prev_kappa = np_zeros((minibatch_size, att_size))
        prev_w = np_zeros((minibatch_size, n_chars))
        bias = args.bias
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
                                            prev_h3, prev_kappa, prev_w, bias,
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
                                        prev_h3, prev_kappa, prev_w, bias,
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
                ii = reconstruct(ex)
                wavfile.write(s, fs, soundsc(ii))
        valid_itr.reset()
        print("Sampling complete, exiting...")
        sys.exit()
    else:
        print("No plotting arguments, starting training mode!")

    X_sym = tensor.tensor3("X_sym")
    X_sym.tag.test_value = X_mb[:cut_len]
    X_mask_sym = tensor.matrix("X_mask_sym")
    X_mask_sym.tag.test_value = X_mb_mask[:cut_len]
    c_sym = tensor.tensor3("c_sym")
    c_sym.tag.test_value = c_mb
    c_mask_sym = tensor.matrix("c_mask_sym")
    c_mask_sym.tag.test_value = c_mb_mask
    bias_sym = tensor.scalar("bias_sym")
    bias_sym.tag.test_value = 0.

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

    cell1 = GRU(input_dim, n_hid, random_state)
    cell2 = GRU(n_hid, n_hid, random_state)
    cell3 = GRU(n_hid, n_hid, random_state)
    params += cell1.get_params()
    params += cell2.get_params()
    params += cell3.get_params()

    v_cell1 = GRU(1, n_v_hid, random_state)
    params += v_cell1.get_params()

    # Use GRU classes only to fork 1 inp to 2 inp:gate pairs
    inp_to_h1 = GRUFork(input_dim, n_hid, random_state)
    inp_to_h2 = GRUFork(input_dim, n_hid, random_state)
    inp_to_h3 = GRUFork(input_dim, n_hid, random_state)
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

    inp_to_v_h1 = GRUFork(1, n_v_hid, random_state)
    params += inp_to_v_h1.get_params()

    h1_to_att_a, h1_to_att_b, h1_to_att_k = make_weights(n_hid, 3 * [att_size],
                                                         random_state)
    params += [h1_to_att_a, h1_to_att_b, h1_to_att_k]

    # Need a , on single results since it always returns a list
    h1_to_outs, = make_weights(n_hid, [n_hid], random_state)
    h2_to_outs, = make_weights(n_hid, [n_hid], random_state)
    h3_to_outs, = make_weights(n_hid, [n_hid], random_state)
    params += [h1_to_outs, h2_to_outs, h3_to_outs]

    # 2 * for mag and phase
    v_outs_to_corr_outs, = make_weights(n_v_hid, [1], random_state)
    corr_outs_to_final_outs, = make_weights(n_hid, [2 * n_density],
                                            random_state)
    params += [v_outs_to_corr_outs, corr_outs_to_final_outs]

    inpt = X_sym[:-1]
    target = X_sym[1:]
    mask = X_mask_sym[1:]
    context = c_sym * c_mask_sym.dimshuffle(0, 1, 'x')

    inp_h1, inpgate_h1 = inp_to_h1.proj(inpt)
    inp_h2, inpgate_h2 = inp_to_h2.proj(inpt)
    inp_h3, inpgate_h3 = inp_to_h3.proj(inpt)

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

    init_x = as_shared(np_zeros((minibatch_size, n_out)))
    srng = RandomStreams(1999)

    def _slice_outs(outs):
        k = n_components
        half = n_out // 2
        outs = outs.reshape((-1, n_density))
        mu = outs[:, 0:half * k].reshape((-1, half, k))
        sigma = outs[:, half * k:2 * half * k].reshape(
            (-1, half, k))
        coeff = outs[:, 2 * half * k:]
        sigma = tensor.exp(sigma - bias_sym) + 1E-6
        coeff = tensor.nnet.softmax(coeff * (1. + bias_sym)) + 1E-6
        return mu, sigma, coeff

    # Used to calculate stopping heuristic from sections 5.3
    u_max = 0. * tensor.arange(c_sym.shape[0]) + c_sym.shape[0]
    u_max = u_max.dimshuffle('x', 'x', 0)
    u_max = tensor.cast(u_max, theano.config.floatX)

    def sample_out_step(x_tm1, v_h1_tm1):
        vinp_h1_t, vgate_h1_t = inp_to_v_h1.proj(x_tm1)
        v_h1_t = v_cell1.step(vinp_h1_t, vgate_h1_t, v_h1_tm1)
        return v_h1_t

    def sample_step(x_tm1, h1_tm1, h2_tm1, h3_tm1, k_tm1, w_tm1, ctx):
        xinp_h1_t, xgate_h1_t = inp_to_h1.proj(x_tm1)
        xinp_h2_t, xgate_h2_t = inp_to_h2.proj(x_tm1)
        xinp_h3_t, xgate_h3_t = inp_to_h3.proj(x_tm1)

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

        out_t = out_t.dimshuffle(1, 0, 'x')

        # vertical scan
        init_v_out = tensor.zeros((out_t.shape[1], n_v_hid))
        v_out_t, updates = theano.scan(
            fn=sample_out_step,
            sequences=[out_t],
            outputs_info=[init_v_out])

        corr_out_t = v_out_t.dot(v_outs_to_corr_outs)
        corr_out_t = corr_out_t[:, :, 0].dimshuffle(1, 0)
        corr_out_t = corr_out_t.dot(corr_outs_to_final_outs)

        split = corr_out_t.shape[-1] // 2
        mag_out_t = corr_out_t[:, :split]
        phase_out_t = corr_out_t[:, split:]
        mu_mag, sigma_mag, coeff_mag = _slice_outs(mag_out_t)
        mu_phase, sigma_phase, coeff_phase = _slice_outs(phase_out_t)
        s_mag = sample_diagonal_gmm(mu_mag, sigma_mag, coeff_mag, srng)
        s_phase = sample_diagonal_gmm(mu_phase, sigma_phase, coeff_phase, srng)
        """
        # Set sample to debug in order to check test values
        s_mag = sample_diagonal_gmm(mu_mag, sigma_mag, coeff_mag, srng,
                                    debug=True)
        s_phase = sample_diagonal_gmm(mu_phase, sigma_phase, coeff_phase, srng,
                                      debug=True)
        """
        s_phase = tensor.mod(s_phase + np.pi, 2 * np.pi) - np.pi
        x_t = tensor.concatenate([s_mag, s_phase], axis=-1)
        return x_t, h1_t, h2_t, h3_t, k_t, w_t, ss_t, sh_t


    n_steps_sym = tensor.iscalar()
    n_steps_sym.tag.test_value = 10
    (sampled, h1_s, h2_s, h3_s, k_s, w_s, stop_s, stop_h), supdates = theano.scan(
        fn=sample_step,
        n_steps=n_steps_sym,
        sequences=[],
        outputs_info=[init_x, init_h1, init_h2, init_h3,
                      init_kappa, init_w, None, None],
        non_sequences=[context])

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

    orig_shapes = outs.shape
    outs = outs.dimshuffle(2, 1, 0)
    # pre project? cutting down to 1 dim really hurts
    outs = outs.reshape((orig_shapes[2], orig_shapes[1] * orig_shapes[0], 1))

    def out_step(x_tm1, v_h1_tm1):
        vinp_h1_t, vgate_h1_t = inp_to_v_h1.proj(x_tm1)
        v_h1_t = v_cell1.step(vinp_h1_t, vgate_h1_t, v_h1_tm1)
        return v_h1_t

    init_v_outs = tensor.zeros((outs.shape[1], n_v_hid))
    v_outs, updates = theano.scan(
        fn=out_step,
        sequences=[outs],
        outputs_info=[init_v_outs])

    corr_outs = v_outs.dot(v_outs_to_corr_outs)
    corr_outs = corr_outs[:, :, 0].reshape((orig_shapes[2], orig_shapes[1],
                                            orig_shapes[0]))
    corr_outs = corr_outs.dimshuffle(2, 1, 0)
    corr_outs = corr_outs.dot(corr_outs_to_final_outs)

    split = corr_outs.shape[-1] // 2
    mag_outs = corr_outs[:, :, :split]
    phase_outs = corr_outs[:, :, split:]

    mu_mag, sigma_mag, coeff_mag = _slice_outs(mag_outs)
    mu_phase, sigma_phase, coeff_phase = _slice_outs(phase_outs)

    target_split = n_out // 2
    mag_target = target[:, :, :target_split]
    phase_target = target[:, :, target_split:]

    mag_cost = diagonal_gmm(
        mag_target, mu_mag, sigma_mag, coeff_mag)
    phase_cost = diagonal_phase_gmm(
        phase_target, mu_phase, sigma_phase, coeff_phase)
    cost = mag_cost + phase_cost

    cost = cost * mask
    cost = cost.sum() / cut_len
    grads = tensor.grad(cost, params)
    grads = gradient_clipping(grads, 10.)

    learning_rate = 1E-4

    opt = adam(params, learning_rate)
    updates = opt.updates(params, grads)

    train_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                      init_h1, init_h2, init_h3, init_kappa,
                                      init_w, bias_sym],
                                     [cost, h1, h2, h3, kappa, w],
                                     updates=updates)
    cost_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                     init_h1, init_h2, init_h3, init_kappa,
                                     init_w, bias_sym],
                                    [cost, h1, h2, h3, kappa, w])
    predict_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                        init_h1, init_h2, init_h3, init_kappa,
                                        init_w, bias_sym],
                                       [corr_outs],
                                       on_unused_input='warn')
    attention_function = theano.function([X_sym, X_mask_sym, c_sym, c_mask_sym,
                                          init_h1, init_h2, init_h3, init_kappa,
                                          init_w],
                                         [kappa, w], on_unused_input='warn')
    sample_function = theano.function([c_sym, c_mask_sym, init_h1, init_h2,
                                       init_h3, init_kappa, init_w, bias_sym,
                                       n_steps_sym],
                                      [sampled, h1_s, h2_s, h3_s, k_s, w_s,
                                       stop_s, stop_h],
                                      updates=supdates)

    checkpoint_dict = {}
    checkpoint_dict["train_function"] = train_function
    checkpoint_dict["cost_function"] = cost_function
    checkpoint_dict["predict_function"] = predict_function
    checkpoint_dict["attention_function"] = attention_function
    checkpoint_dict["sample_function"] = sample_function

    print("Beginning training loop")
    train_mb_count = 0
    valid_mb_count = 0
    start_epoch = 0
    monitor_frequency = 1000 // minibatch_size
    overall_train_costs = []
    overall_valid_costs = []

    if args.cont is not None:
        continue_path = args.cont
        if not os.path.exists(continue_path):
            raise ValueError("Continue model %s, path not "
                             "found" % continue_path)
        saved_checkpoint = load_checkpoint(continue_path)
        trained_weights = get_values_from_function(
            saved_checkpoint["train_function"])
        set_shared_variables_in_function(train_function, trained_weights)
        try:
            overall_train_costs = saved_checkpoint["overall_train_costs"]
            overall_valid_costs = saved_checkpoint["overall_valid_costs"]
            start_epoch = len(overall_train_costs)
        except KeyError:
            print("Key not found - model structure may have changed.")
            print("Continuing anyways - statistics may not be correct!")

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
            if len(X_mb[start:stop]) < cut_len:
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
            bias = 0.  # No bias in training
            rval = function(X_mb[start:stop],
                            X_mb_mask[start:stop],
                            c_mb, c_mb_mask,
                            prev_h1, prev_h2, prev_h3, prev_kappa, prev_w, bias)
            current_cost = rval[0]
            prev_h1, prev_h2, prev_h3 = rval[1:4]
            prev_h1 = prev_h1[-1]
            prev_h2 = prev_h2[-1]
            prev_h3 = prev_h3[-1]
            prev_kappa = rval[4][-1]
            prev_w = rval[5][-1]
        partial_costs.append(current_cost)
        return partial_costs

    for e in range(start_epoch, start_epoch + n_epochs):
        train_costs = []
        try:
            while True:
                partial_train_costs = _loop(train_function, train_itr)
                train_costs.append(np.mean(partial_train_costs))
                if train_mb_count % monitor_frequency == 0:
                    print("starting train mb %i" % train_mb_count)
                    print("current epoch mean cost %f" % np.mean(train_costs))
	        if np.isnan(train_costs[-1]) or np.isinf(train_costs[-1]):
		    print("Invalid cost detected at epoch %i" % e)
		    raise ValueError("Exiting...")
                train_mb_count += 1
        except StopIteration:
            valid_costs = []
            try:
                while True:
                    partial_valid_costs = _loop(cost_function, valid_itr)
                    valid_costs.append(np.mean(partial_valid_costs))
                    if valid_mb_count % monitor_frequency == 0:
                        print("starting valid mb %i" % valid_mb_count)
                        print("current validation mean cost %f" % np.mean(
                            valid_costs))
                    valid_mb_count += 1
            except StopIteration:
                pass
            mean_epoch_train_cost = np.mean(train_costs)
            mean_epoch_valid_cost = np.mean(valid_costs)
            overall_train_costs.append(mean_epoch_train_cost)
            overall_valid_costs.append(mean_epoch_valid_cost)
            checkpoint_dict["overall_train_costs"] = overall_train_costs
            checkpoint_dict["overall_valid_costs"] = overall_valid_costs
            script = os.path.realpath(__file__)
            print("script %s" % script)
            print("epoch %i complete" % e)
            print("epoch mean train cost %f" % mean_epoch_train_cost)
            print("epoch mean valid cost %f" % mean_epoch_valid_cost)
            print("overall train costs %s" % overall_train_costs[-5:])
            print("overall valid costs %s" % overall_valid_costs[-5:])
            if ((e % checkpoint_every_n) == 0) or (e == (n_epochs - 1)):
                print("Checkpointing...")
                checkpoint_save_path = "model_checkpoint_%i.pkl" % e
                weights_save_path = "model_weights_%i.npz" % e
                save_checkpoint(checkpoint_save_path, checkpoint_dict)
                save_weights(weights_save_path, checkpoint_dict)
