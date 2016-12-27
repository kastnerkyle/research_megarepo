# A reduction of the post by Yoav Goldberg to a script
# https://gist.github.com/yoavg/d76121dfde2618422139
# Author: Kyle Kastner
# License: BSD 3-Clause

# Fun alternate settings
# Download kjv.txt from http://www.ccel.org/ccel/bible/kjv.txt
# python markov_lm.py kjv.txt 5 1.
# Snippet:
#  Queen ording found Raguel: I kill.
#  THROUGH JESUS OF OUR BRETHREN, AND PEACE,
#
#  NUN.

from collections import defaultdict, Counter
import os
import sys
import numpy as np

# Reduce memory on python 2
if sys.version_info < (3, 0):
    range = xrange


def train_char_lm(fname, order=4, temperature=1.0):
    data = file(fname).read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data

    for i in range(len(data) - order):
        history, char = data[i:i + order], data[i + order]
        lm[history][char] += 1

    def normalize(counter):
        # Use a proper softmax with temperature
        t = temperature
        ck = counter.keys()
        cv = counter.values()
        # Keep it in log space
        s = float(sum([pi for pi in cv]))
        # 0 to 1 to help numerical issues
        p = [pi / s for pi in cv]
        # log_space
        p = [pi / float(t) for pi in p]
        mx = max(p)
        # log sum exp
        s_p = mx + np.log(sum([np.exp(pi - mx) for pi in p]))
        # Calculate softmax in a hopefully more stable way
        # s(xi) = exp ^ (xi / t) / sum exp ^ (xi / t)
        # log s(xi) = log (exp ^ (xi / t) / sum exp ^ (xi / t))
        # log s(xi) = log exp ^ (xi / t) - log sum exp ^ (xi / t)
        # with pi = xi / t
        # with s_p = log sum exp ^ (xi / t)
        # log s(xi) = pi - s_p
        # s(xi) = np.exp(pi - s_p)
        p = [np.exp(pi - s_p) for pi in p]
        return [(ci, pi) for ci, pi in zip(ck, p)]

    outlm = {hist: normalize(chars) for hist, chars in lm.iteritems()}
    return outlm


def generate_letter(lm, history, order, random_state):
    history = history[-order:]
    dist = lm[history]
    x = random_state.rand()
    for c, v in dist:
        x = x - v
        if x <= 0:
            return c
    # randomize choice if it all failed
    li = list(range(len(dist)))
    random_state.shuffle(li)
    c, _ = dist[li[0]]
    return c


def generate_text(lm, order, n_letters=1000):
    history = "~" * order
    out = []
    random_state = np.random.RandomState(2145)
    for i in range(n_letters):
        c = generate_letter(lm, history, order, random_state)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)


if __name__ == "__main__":

    default_order = 6
    default_temperature = 1.0
    default_fpath = "shakespeare_input.txt"

    if len(sys.argv) > 1:
        fpath = sys.argv[1]
        if not os.path.exists(fpath):
            raise ValueError("Unable to find file at %s" % fpath)
    else:
        fpath = default_fpath
        if not os.path.exists(fpath):
            raise ValueError("Default shakespeare file not found!"
                             "Get the shakespeare file from http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
                             "Place at %s" % fpath)
    if len(sys.argv) > 2:
        order = int(sys.argv[2])
    else:
        order = default_order

    if len(sys.argv) > 3:
        temperature = float(sys.argv[3])
    else:
        temperature = default_temperature

    lm = train_char_lm(fpath, order=order,
                       temperature=temperature)
    print(generate_text(lm, order))
