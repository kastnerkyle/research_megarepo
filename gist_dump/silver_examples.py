import numpy as np

# written out matrix for classwork example in Silver lecture 2

# square, rows sum to 1 (stochastic matrix)
# class1, class2, class3, pass, pub, fb, sleep
transition_p = np.array([[0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
                         [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
                         [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

rewards = np.array([-2., -2., -2., 10., 1., -1., 0.])
discount = 0.5

# array of arrays so indexing is easy
fixed_chains = np.array([np.array([0, 1, 2, 3, 6]),
                         np.array([0, 5, 5, 0, 1, 6]),
                         np.array([0, 1, 2, 4, 1, 2, 4, 6]),
                         np.array([0, 5, 5, 0, 1, 2, 4, 0, 5, 5, 5, 0, 1, 2, 4,
                                   2, 6]),
                         ])

v1_1 = np.sum(rewards[fixed_chains[0]] *
              discount ** np.arange(len(fixed_chains[0])))
v1_2 = np.sum(rewards[fixed_chains[1]]
              * discount ** np.arange(len(fixed_chains[1])))
v1_3 = np.sum(rewards[fixed_chains[2]]
              * discount ** np.arange(len(fixed_chains[2])))
v1_4 = np.sum(rewards[fixed_chains[3]]
              * discount ** np.arange(len(fixed_chains[3])))
print(v1_1, v1_2, v1_3, v1_4)


def solve_bellman_value_equation(rewards, transitions, discount):
    # closed form version
    inv = np.linalg.pinv(np.eye(transitions.shape[0]) - discount * transitions)
    return inv.dot(rewards[:, None])

print("Discount 0.9")
print(solve_bellman_value_equation(rewards, transition_p, 0.9))
print("Discount 1.0")
print(solve_bellman_value_equation(rewards, transition_p, 1.0))

# ???