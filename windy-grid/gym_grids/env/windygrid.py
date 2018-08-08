"""
Windy Grid World system implemented by Utkarsh Saxena
"""

import gym
from gym import spaces, logger
import numpy as np
from gym.envs.classic_control import rendering
import random


class WindyGridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self, rows=7, col=10):
        self.rows = rows
        self.col = col
        self.wind = np.random.randint(3, size=col) - 1
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.action_space = spaces.Discrete(4)
        self.moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.start_state = (rows // 2, 0)
        self.end_state = (rows // 2, col // 2 + 2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.rows),
            spaces.Discrete(self.col)
        ))
        self.steps_beyond_done = None
        self.state = self.start_state
        self.done = False
        self.path = []
        self.viewer = None
        self.reset()
        random.seed(0)

    def bound(self, x, y):
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x >= self.rows:
            x = self.rows - 1
        if y >= self.col:
            y = self.col - 1
        return x, y

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        dx, dy = self.moves[action]
        x, y = self.state[0] + dx, self.state[1] + dy
        x, y = self.bound(x, y)
        x += self.wind[self.state[1]]  # the wind (of older state) effects you when jump away from it
        x, y = self.bound(x, y)
        self.state = (x, y)
        if self.state == self.end_state:
            self.done = True

        if not self.done:
            reward = -1.0
        elif self.steps_beyond_done is None:
            # Goal Reached just now
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- any further steps are "
                    "undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        self.path.append(self.state)
        return self.state, reward, self.done, {}

    def reset(self):
        self.done = False
        self.state = self.start_state
        self.steps_beyond_done = None
        self.path = [self.state]
        self.path_colour = [random.random(), random.random(), random.random()]
        return self.state

    def render(self, mode='human'):
        screen_width = 900
        screen_height = 800

        world_width = self.col
        world_height = self.rows
        scale_w = screen_width / world_width
        scale_h = screen_height / world_height

        def box(state):
            x, y = state
            x1, y1, x2, y2 = y * scale_w, x * scale_h, (y + 1) * scale_w, (x + 1) * scale_h
            return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)
            for row in range(self.rows):
                line = rendering.Line((0, row * scale_h), (screen_width, row * scale_h))
                self.viewer.add_geom(line)
            for col in range(self.col):
                line = rendering.Line((col * scale_w, 0), (col * scale_w, screen_height))
                self.viewer.add_geom(line)
            self.translation = rendering.Transform()
            robot = rendering.make_circle(min(scale_w, scale_h) / 2)
            robot.add_attr(self.translation)
            start = rendering.FilledPolygon(box(self.start_state))
            start.set_color(0, 1, 0)
            finish = rendering.FilledPolygon(box(self.end_state))
            finish.set_color(0, 0, 1)
            self.viewer.add_geom(start)
            self.viewer.add_geom(finish)
            self.viewer.add_geom(robot)

        x = self.state[0] * scale_h + scale_h / 2
        y = self.state[1] * scale_w + scale_w / 2
        path = rendering.make_polyline([((y + 0.5) * scale_w, (x + 0.5) * scale_h) for (x, y) in self.path])
        path.set_color(*self.path_colour)
        path.set_linewidth(5.5)
        self.viewer.add_geom(path)
        self.translation.set_translation(y, x)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
