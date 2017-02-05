import random
import math
import cPickle
    
def generate_states():
    for l1 in xrange(21):
        for l2 in xrange(21):
            yield (l1, l2)

def actions_for_state(state):
    l1, l2 = state
    for i in xrange(1, min(l1, 5) + 1):
        if l2 + i >= 20:
            break
        yield i
    for j in xrange(1, min(l2, 5) + 1):
        if l1 + j >= 20:
            break
        yield -j
    yield 0

def factorial(actual):
    if actual == 0:
        return 1
    return reduce(lambda x, y: x * y, xrange(1, actual + 1), 1)

def poisson(expected, actual):
    return math.pow(expected, actual) / float(factorial(actual)) * math.exp(-float(expected))

def truncated_poisson(expected, actual, maximum):
    if actual >= maximum:
        return 1.0 - sum((poisson(expected, n) for n in xrange(maximum)))
    else:
        return poisson(expected, actual)

def possibilities_loc(current_cars, expected_rentals=3, expected_returns=3, max_cars=20):
    dist = {}
    max_rentals = current_cars
    for rentals in xrange(0, max_rentals + 1):
        max_returns = 20 - (current_cars - rentals)
        for returns in xrange(0, max_returns + 1):
            p_rentals = truncated_poisson(expected_rentals, rentals, max_rentals)
            p_returns = truncated_poisson(expected_returns, returns, max_returns)
            dist[(rentals, returns)] = p_rentals * p_returns
    return dist

def possibilities_grid(current_state, action, values, discount=0.9):
    current_cars_l1, current_cars_l2 = current_state
    current_cars_l1 -= action
    current_cars_l2 += action
    possibilities_l1 = possibilities_loc(current_cars_l1, 3, 3)
    possibilities_l2 = possibilities_loc(current_cars_l2, 4, 2)
    bellman_value = 0.0
    for (l1_rentals, l1_returns), l1_proba in possibilities_l1.items():
        for (l2_rentals, l2_returns), l2_proba in possibilities_l2.items():
            proba = l1_proba * l2_proba
            reward = 10 * (l1_rentals + l2_rentals) - abs(action) * 2
            state = (current_cars_l1 - l1_rentals + l1_returns, current_cars_l2 - l2_rentals + l2_returns)
            bellman_value += proba * (reward + discount * values[state])
    return bellman_value

def policy_iteration(model_filename, discount=0.9, delta_thresh=0.001):
    states = set(generate_states())
    # Initialization:
    try:
        print "Attempting to load model..."
        with open(model_filename, "rb") as f:
            values, policy = cPickle.load(f)
        print "Loaded."
    except:
        print "Couldn't load model, starting from scratch."
        values = {state: random.random() for state in states}
        policy = {state: 0 for state in states}

    def checkpoint():
        with open(model_filename, "wb") as f:
            cPickle.dump([values, policy], f)
    
    stable = False
    while not stable:
        # Policy evaluation:
        print "Starting policy evaluation"
        while True:
            delta = 0
            for state in states:
                temp = values[state]
                values[state] = possibilities_grid(state, policy[state], values, discount=discount)
                delta = max(delta, abs(temp - values[state]))
            checkpoint()
            print "DELTA: %f" % delta
            if delta < delta_thresh:
                break
        print "Finished policy evaluation"
                
        # Policy improvement:
        print "Starting policy improvement"
        stable = True
        for state in states:
            temp = policy[state]
            action_value = lambda action: possibilities_grid(state, action, values, discount=discount)
            policy[state] = max(((action_value(action), action) for action in actions_for_state(state)))[1]
            if policy[state] != temp:
                stable = False
        checkpoint()
        print "Finished policy improvement"
    return policy

import matplotlib.pyplot as plt
import numpy as np
def heatmap(Z, colorlabel=None):
    CS = plt.pcolor(Z)
    # put the major ticks at the middle of each cell, notice "reverse" use of dimension
    ax = plt.gca()
    ax.set_yticks(np.arange(Z.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(Z.shape[1])+0.5, minor=False)
    ax.set_xticklabels(np.arange(0, 21, 1), minor=False)
    ax.set_yticklabels(np.arange(0, 21, 1), minor=False)
    plt.xlim([0,20])
    plt.ylim([0,20])
    cbar = plt.colorbar(CS)
    if colorlabel is not None:
        cbar.ax.set_ylabel(colorlabel)

def visualize(model_filename):
    states = set(generate_states())
    print "Attempting to load model..."
    with open(model_filename, "rb") as f:
        values, policy = cPickle.load(f)
    print "Loaded."
    
    plt.figure(1)

    Z = np.array([[policy[(x, y)] for x in xrange(21)] for y in xrange(21)])
    plt.subplot(211)
    plt.title("Policy")
    heatmap(Z, colorlabel='net cars L1 -> L2')
    plt.xlabel("Cars in L1")
    plt.ylabel("Cars in L2")
    plt.gca().set_aspect('equal')
    plt.gca().autoscale('tight')

    plt.subplot(212)
    Z = np.array([[values[(x, y)] for x in xrange(21)] for y in xrange(21)])
    heatmap(Z, colorlabel='value')
    plt.title("Value")
    plt.xlabel("Cars in L1")
    plt.ylabel("Cars in L2")
    plt.gca().set_aspect('equal')
    plt.gca().autoscale('tight')

    plt.show()

import sys
if __name__ == "__main__":
    assert(len(sys.argv) == 3)
    if sys.argv[1] == "policy":
        policy_iteration(sys.argv[2])
    elif sys.argv[1] == "vis":
        visualize(sys.argv[2])
    else:
        print "Unrecognized command"
