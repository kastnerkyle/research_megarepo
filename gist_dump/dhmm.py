# (C) Kyle Kastner, June 2014
# License: BSD 3 clause

import numpy as np


class dhmm:
    def __init__(self, n_states, initial_prob=None,
                 n_iter=100, random_seed=1999):
        # Initial state probabilities p(s_0)=pi[s_0].
        # Transition matrix p(s_j|s_i)=t[s_i][s_j]
        # Emission matrix p(o_i|s_i)=e[s_i][o_i]
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        if initial_prob is None:
            self.initial_prob = np.array([1. / n_states] * n_states)
        else:
            self.initial_prob = initial_prob

    def _forward(self, X, transition_prob, emission_prob):
        # Eventually convert to logspace
        n_states, n_steps = emission_prob.shape
        forward = np.zeros((n_states, n_steps + 1))
        log_likelihood = 0.
        forward[:, 0] = self.initial_prob
        for i in range(n_steps):
            f_i = forward[:, i]
            e_i = emission_prob[:, X[i]]
            forward[:, i + 1] = np.dot(f_i[None], transition_prob) * e_i
            forward_sum = np.sum(forward[:, i + 1])
            forward[:, i + 1] = forward[:, i + 1] / forward_sum
            log_likelihood += np.log(forward_sum)
        return log_likelihood, forward

    def _backward(self, X, transition_prob, emission_prob):
        # Eventually convert to logspace
        n_states, n_steps = emission_prob.shape
        backward = np.zeros((n_states, n_steps + 1))
        backward[:, -1] = 1.
        for i in range(n_steps, 0, -1):
            b_i = backward[:, i]
            e_p = emission_prob[:, X[i - 1]]
            backward[:, i - 1] = np.dot(transition_prob * e_p,
                                        b_i[None].T).ravel()
            backward[:, i - 1] = backward[:, i - 1] / np.sum(backward[:, i - 1])
        return backward

    def _baum_welch(self, X, transition_prob, emission_prob, initial_prob=None):
        for i in range(len(X)):
            X_i = X[i]
            n_states, n_steps = emission_prob.shape
            old_transition = transition_prob
            old_emission = emission_prob
            transition = np.ones_like(old_transition)
            emission = np.ones_like(old_emission)
            ll, forward = self._forward(X_i, old_transition, old_emission)
            backward = self._backward(X_i, old_transition, old_emission)
            probs = forward * backward
            probs = probs / np.sum(probs, axis=0)
            theta = np.zeros((n_states, n_states, n_steps))
            for a in range(n_states):
                for b in range(n_states):
                    for t in range(n_steps):
                        theta[a, b, t] = (forward[a, t] *
                                          backward[b, t + 1] *
                                          old_transition[a, b] *
                                          old_emission[b, X_i[t]])
            for a in range(n_states):
                for b in range(n_states):
                    transition[a, b] = np.sum(
                        theta[a, b, :]) / np.sum(probs[a, :])
            transition = transition / np.sum(transition, axis=1)
            for a in range(n_states):
                for t in range(n_steps):
                    right_ind = np.array(np.where(X_i == t)) + 1
                    emission[a, t] = np.sum(probs[a, right_ind]) / np.sum(
                        probs[a, 1:])
            emission = emission / np.sum(emission, axis=1)[:, None]
        return transition, emission

    def _setup(self, X):
        # Samples, Time, Features
        n_steps = X.shape[1]
        n_states = self.n_states
        self.initial_prob_ = np.ones((n_states,))
        self.initial_prob_ /= np.sum(self.initial_prob_)
        self.transition_prob_ = np.ones((n_states, n_states))
        self.transition_prob_ /= np.sum(self.transition_prob_, axis=1)
        self.emission_prob_ = np.ones((n_states, n_steps))
        self.transition_prob_ /= np.sum(self.emission_prob_, axis=1)

    def fit(self, X, y=None):
        self._setup(X)
        for n in range(self.n_iter):
            self.partial_fit(X)

    def partial_fit(self, X):
        if not hasattr(self, 'transition_prob_'):
            self._setup(X)
        t = self.transition_prob_
        e = self.emission_prob_
        t, e = self._baum_welch(X, t, e)
        self.transition_prob_ = t
        self.emission_prob_ = e

    def score(self, X):
        scores = []
        for i in range(len(X)):
            X_i = X[i]
            n_states, n_steps = self.emission_prob_.shape
            ll, forward = self._forward(X_i, self.transition_prob_,
                                        self.emission_prob_)
            scores.append(ll)
        return np.array(scores)

if __name__ == "__main__":
    n_steps = 50
    rs = np.random.RandomState(1999)
    X1 = rs.randn(1, n_steps)
    X1[X1 > 0] = 1.
    X1[X1 <= 0] = 0
    X2 = rs.rand(1, n_steps)
    X2[X2 > .15] = 1.
    X2[X2 <= 0.85] = 0.

    m = dhmm(2)
    m.fit(X1)
    print(m.score(X1))
    print(m.score(X2))

    m = dhmm(2)
    m.fit(X2)
    print(m.score(X1))
    print(m.score(X2))

    m = dhmm(2)
    for i in range(100):
        m.partial_fit(X2)
    print(m.score(X1))
    print(m.score(X2))
