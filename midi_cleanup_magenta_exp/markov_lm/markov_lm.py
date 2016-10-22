from collections import defaultdict, Counter
import os
import sys
import numpy as np

# Reduce memory on python 2
if sys.version_info < (3, 0):
    range = xrange


def train_char_lm(fname, order=4, temperature=1.):
    data = file(fname).read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data) - order):
        history, char = data[i:i + order], data[i + order]
        lm[history][char] += 1

    def normalize(counter):
        t = temperature
        mx = max(counter.values())
        # proper softmax
        # recale between -inf and 0 for some stability
        lm_norm = [(c, cnt - mx) for c, cnt in counter.iteritems()]
        # calculate temperature in a more stable way by staying in log space
        lm_exp = [(c, p / t) for c, p in lm_norm]
        # logsumexp, even though mx *should* be 0
        mx = max([p for _, p in lm_exp])
        lm_sum = mx + np.log(sum([np.exp(p - mx) for _, p in lm_exp]))
        # base softmax (denoted s) calculation is
        # s(xi) = exp ^ (xi / t) / sum exp ^ (xi / t)
        # log of that is
        # log s(xi) = log (exp ^ xi / t / sum exp ^ xi / t)
        # log s(xi) = log exp ^ xi / t - log sum exp ^ xi / t
        # Finally we calculate
        # log s(xi) = xi / t - log sum exp ^ xi / t
        # Applying exp again
        # exp log s(xi) = exp ^ (xi / t - log sum exp ^ xi / t)
        # with variable p = xi / t and lm_sum = log sum exp p ...
        # s(xi) = exp ^ (p - lm_sum)
        lm = [(c, np.exp(p - lm_sum)) for c, p in lm_exp]
        return lm
    outlm = {hist: normalize(chars) for hist, chars in lm.iteritems()}
    return outlm


def generate_letter(lm, history, order, random_state):
    history = history[-order:]
    dist = lm[history]
    x = random_state.rand()
    # spaces are wildly overrepresented
    if history[-1] == " ":
        no_space = True
    else:
        no_space = False
    for c, v in dist:
        x = x - v
        if x <= 0:
            if no_space and c == " ":
                continue
            else:
                return c
    return c


def generate_text(lm, order, n_letters=1000):
    history = "~" * order
    out = []
    random_state = np.random.RandomState(1999)
    for i in range(n_letters):
        c = generate_letter(lm, history, order, random_state)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)


if __name__ == "__main__":
    default_order = 6
    default_length = 1000
    default_temperature = 1.
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = "shakespeare_input.txt"

    if len(sys.argv) > 2:
        order = int(sys.argv[2])
    else:
        order = default_order

    if len(sys.argv) > 3:
        temperature = float(sys.argv[3])
    else:
        temperature = default_temperature

    if len(sys.argv) > 4:
        length = int(sys.argv[4])
    else:
        length = default_length


    if not os.path.exists(fname):
        if len(sys.argv) <= 1:
            raise ValueError("Default file not found!",
                             "Get it from http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt")
        else:
            raise ValueError("File %s not found!" % fname)

    lm = train_char_lm(fname, order, temperature)
    print(generate_text(lm, order, n_letters=length))
