import gym
import random
import time
from collections import defaultdict

random.seed(1)

INF = 100000000000000


def flip_coin(eps):
    return 1 if random.random() < eps else 0


def get_q_value(state, a, q_value):
    if (state, a) in q_value:
        return q_value[(state, a)]
    return -INF


def greedy(q_value, state, actions):
    a_ = 0
    for a in actions:
        if get_q_value(state, a, q_value) > get_q_value(state, a_, q_value):
            a_ = a
    if get_q_value(state, a_, q_value) == -INF:
        a_ = random.choice(actions)
    return a_


def epsilon_greedy(q_value, state, actions, epsilon):
    if flip_coin(epsilon):
        return random.choice(actions)
    else:
        # Choose a greedy strategy
        return greedy(q_value, state, actions)


def show_solution(env, q_value, actions, epsilon=0.0, wait=False, delay=0.1, debug=False):
    done = False
    state = env.reset()
    print("showing solution for eps = {0:.4f}".format(epsilon))
    if wait:
        print("Want to see the solution ?")
        x = input()
    limit = 100
    while not done and limit > 0:
        action = epsilon_greedy(q_value, state, actions, epsilon)
        if debug:
            print("action = {}".format(action))
            print(["({0},{1:.2f}) ".format(a, get_q_value(state, a, q_value)) for a in actions])
            print(state, action)
        ns, reward, done, info = env.step(action)
        env.render()
        if ns == state:
            break
        state = ns
        time.sleep(delay)
        limit -= 1


def monte_carlo_epsilon_greedy(env, episodes=1000000, epsilon=0.80):
    print(env.env.wind)
    actions = [0, 1, 2, 3]
    q_value = defaultdict(float)
    n = defaultdict(int)
    for _ in range(episodes):
        start = state = env.reset()
        done = False
        history = []
        if _ > 0.05 * episodes:
            epsilon = 1 / (1 + _)
        while not done:
            action = epsilon_greedy(q_value, state, actions, epsilon)
            n_state, reward, done, info = env.step(action)
            history.append(((state, action), reward))
            state = n_state
        assert state == env.env.end_state
        if _ % 500 == 0:
            print("Progress = {0:.3f}%, eps = {1:.3f}, value[start] = {2:.4f}, last episode length = {3},"
                  " correct answer={4}".
                  format(_ * 100 / episodes,
                         epsilon,
                         get_q_value(start, greedy(q_value, start, actions), q_value),
                         len(history), env.env.correct_answer))

        G = 0
        last_visit_only = {}
        for ((state, action), reward) in reversed(history):
            if (state, action) in last_visit_only:
                # Forget the cycle. Only true for negative reward cycle (as in this case)
                G = last_visit_only[(state, action)]
            else:
                # First visit to (s,a)
                G += reward
                n[(state, action)] += 1
                q_value[(state, action)] += (G - q_value[(state, action)]) / n[(state, action)]
                last_visit_only[(state, action)] = G

    # Show Greedy Solution
    show_solution(env, q_value, actions, epsilon=0, delay=1.5)


def sarsa(env):
    print(env.env.wind)
    raise NotImplementedError


if __name__ == '__main__':
    import gym_grids

    e = gym.make('Windy-Grid-v0')
    e.reset()
    e._max_episode_steps = 2000000
    monte_carlo_epsilon_greedy(e)
