# MIT License
#
# Copyright (C) IBM Corporation 2018, 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Gym env for https://github.ibm.com/Roland-Everett-Fall/unsupervised_htn/blob/master/simulators/construct.py
'''

import numpy as np
import random
import math
import operator
import gym
from gym import spaces, logger
import cv2
from matplotlib import pyplot as plt
GRID_DIM = (8, 8)
HOUSE_IMAGE = [[None, 'red_tile', 'red_tile', 'red_tile', None],
               ['red_tile', 'red_tile', 'red_tile', 'red_tile', 'red_tile'],
               [None, 'brick', 'brick', 'brick', None],
               [None, 'brick', 'door', 'brick', None]]

FOUNDATION_IMAGE = [[None, None, None, None, None],
                    [None, None, None, None, None],
                    [None, None, None, None, None],
                    [None, 'door', 'door', 'door', None]]


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_rgb_matrix(rgb_matrix, m=15):
    rows = rgb_matrix.shape[0]
    cols = rgb_matrix.shape[1]

    # rgb_array = np.ones_like(rgb_matrix).astype('uint8') * 50
    rgb_array = np.ones((rows * m, cols * m, 3)) * 50
    # pygame.draw.rect(screen, pygame.Color(50, 50, 50),
    # pygame.Rect(0 + x_offset, 0, 800, 600))

    for x in range(0, rows):
        for y in range(0, cols):
            r = int(rgb_matrix[y, x, 0])
            g = int(rgb_matrix[y, x, 1])
            b = int(rgb_matrix[y, x, 2])
            if r < 0:
                r = 0
            elif r > 255:
                r = 255
            if b < 0:
                b = 0
            elif b > 255:
                b = 255
            if g < 0:
                g = 0
            elif g > 255:
                g = 255

            rgb_array[y * m:(y + 1) * m, x * m:(x + 1) * m, 0] = r
            rgb_array[y * m:(y + 1) * m, x * m:(x + 1) * m, 1] = g
            rgb_array[y * m:(y + 1) * m, x * m:(x + 1) * m, 2] = b

            # color = pygame.Color(r, g, b)
    return rgb_array


class ConstructBaseEnv(gym.Env):
    def __init__(self, max_steps=400):
        self.action_space = spaces.Discrete(10)
        self.observation_space = \
            spaces.Box(low=0, high=255, shape=(GRID_DIM[0], GRID_DIM[1], 3))
        self.house_image = \
            [[None, 'red_tile', 'red_tile', 'red_tile', None],
             ['red_tile', 'red_tile', 'red_tile', 'red_tile', 'red_tile'],
             [None, 'brick', 'brick', 'brick', None],
             [None, 'brick', 'door', 'brick', None]]
        self.max_steps = max_steps
        self.reset()

    def seed(self, s):
        pass

    def reset(self):
        self.grid = {}
        self.steps = 0
        for i in range(GRID_DIM[0]):
            for j in range(GRID_DIM[1]):
                self.grid[(i, j)] = 'empty'

        self.regions = np.zeros(GRID_DIM)

        rows = GRID_DIM[0]
        cols = GRID_DIM[1]

        self.t = 0
        self.is_subgoal_state_indicator = False
        self.view_subgoals = False

        '''
        self.centroids={}
        self.region_sizes={}
        self.create_regions()
        '''
        for i in range(self.regions.shape[0]):
            for j in range(self.regions.shape[1]):
                self.grid[(i, j)] = 'land'
        self.checkpoints = []

        # print("size: ",smallest,largest)

        self.x = random.randint(0, rows - 1)
        self.y = random.randint(0, cols - 1)

        # self.reset_if_invalid()
        site = (random.randint(1, 2), random.randint(1, 2))
        self.chosen_sites = [site]

        self.goal_mask = self.grid.copy()
        for chosen_site in self.chosen_sites:
            x = chosen_site[0]
            y = chosen_site[1]
            self.apply_mask(x, y, FOUNDATION_IMAGE, self.goal_mask)
            self.checkpoints.insert(0, self.goal_mask.copy())
            self.apply_mask(x, y, HOUSE_IMAGE, self.goal_mask)

            self.checkpoints.insert(0, self.goal_mask.copy())

        return self.get_rgb_matrix()

    def export_state(self):
        return (self.grid, self.checkpoints, self.goal_mask)

    def load_state(self, grid, checkpoints, goal_mask):
        self.grid = grid.copy()
        self.checkpoints = list(checkpoints)
        self.goal_mask = goal_mask.copy()

    def apply_mask(self, x, y, image, mask):

        h = len(image[0])
        w = len(image)

        if x + w < GRID_DIM[0] and y + h < GRID_DIM[1]:
            for i in range(0, w):
                for j in range(0, h):
                    if image[i][j] is not None:
                        mask[(x + i, y + j)] = image[i][j]

    def choose_house_site(self):
        bad_site = True
        h = len(HOUSE_IMAGE[0])
        w = len(HOUSE_IMAGE)
        x = 0
        y = 0
        count = 0
        while bad_site:
            x = self.target_x + random.randint(-10, 10)
            y = self.target_y + random.randint(-10, 10)

            bad_site = self.near_water_or_site(x, y, 4)
            bad_site = bad_site or x + w >= GRID_DIM[0] or y + h >= GRID_DIM[1]
            count += 1

            if count > 100:
                return False

        # self.world.grid[(x,y)]='red_tile'
        self.chosen_sites.append((x, y))
        return True

    def near_water_or_site(self, x, y, distance):

        if x < 3 or x >= GRID_DIM[0] - 3:
            return True
        if y < 3 or y >= GRID_DIM[1] - 3:
            return True

        for i in range(max(0, x - distance),
                       min(GRID_DIM[0], x + distance + 1)):
            for j in range(max(0, y - distance),
                           min(GRID_DIM[1], y + distance + 1)):
                if self.grid[(i, j)] == 'water':
                    return True
                for (site_x, site_y) in self.chosen_sites:
                    if site_x == i and site_y == j:
                        return True

    def reset_if_invalid(self):

        reset = False
        for centroid in enumerate(self.centroids.items()):
            if centroid[0] == -1:
                reset = True

        if reset:
            print("reset")
            self.__init__()

    def step(self, action):
        # there are 10 actions
        # 0 -> up
        # 1 -> down
        # 2 -> left
        # 3 -> right
        # 4 -> brick
        # 5 -> door
        # 6 -> red_tile
        # 7 -> bridge
        # 8 -> land
        # 9 -> None
        terminated = False

        rows = GRID_DIM[0]
        cols = GRID_DIM[1]

        self.x = max(self.x, 0)
        self.y = max(self.y, 0)

        drigCopy = self.grid.copy()

        # print(action)
        try:
            if action == 0:
                if self.y + \
                        1 < cols and self.grid[(self.x, self.y + 1)] != 'water':
                    self.y += 1

            if action == 1:
                if self.y - \
                        1 >= 0 and self.grid[(self.x, self.y - 1)] != 'water':
                    self.y -= 1

            if action == 2:
                if self.x - \
                        1 >= 0 and self.grid[(self.x - 1, self.y)] != 'water':
                    self.x -= 1

            if action == 3:
                if self.x + \
                        1 < rows and self.grid[(self.x + 1, self.y)] != 'water':
                    self.x += 1

            if action == 4:
                if self.y - \
                        1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                    self.grid[(self.x, self.y - 1)] = 'brick'

            if action == 5:
                if self.y - \
                        1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                    self.grid[(self.x, self.y - 1)] = 'door'

            if action == 6:
                if self.y - \
                        1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                    self.grid[(self.x, self.y - 1)] = 'red_tile'

            if action == 7:
                pass  # self.build_bridge()

            if action == 8:
                if self.y - \
                        1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                    self.grid[(self.x, self.y - 1)] = 'land'

            if action == 9:
                pass
        except BaseException:
            print('Error in action computation.')
            pass
            self.grid.copy = drigCopy.copy()

        self.t += 1

        state = self.get_rgb_matrix()

        rew = -0.1 + self.num_house() * 100

        self.steps += 1

        terminated = self.is_goal_state() or (self.steps >= self.max_steps)

        return state, rew, terminated, {}

    def render(self):
        rgb_matrix = self.get_rgb_matrix()
        # cv2.imshow('simulate', cv2.resize(draw_rgb_matrix(rgb_matrix).astype('uint8')[:,:,::-1],
        #                                   (256, 256), interpolation=cv2.INTER_LANCZOS4))
        cv2.imshow('simulate',
                   draw_rgb_matrix(rgb_matrix).astype('uint8')[:, :, ::-1])
        cv2.waitKey(10)
        # plt.imshow(cv2.resize(draw_rgb_matrix(rgb_matrix), (256, 256)))
        # plt.pause(0.01)

    def match(self, x, y, c):
        if self.grid[x, y] == c or self.grid[x, y] - \
                4 == c or self.grid[x, y] == c - 4:
            return True
        else:
            return False

    def get_rgb_matrix(self, scale=1):
        rows = GRID_DIM[0]
        cols = GRID_DIM[1]
        scale = int(scale)
        rgb_matrix = np.zeros((rows * scale, cols * scale, 3))

        for x_g in range(0, rows):
            for y_g in range(0, cols):
                val = self.grid[(x_g, y_g)]
                x = scale * x_g
                y = scale * y_g
                if val == 'empty':
                    rgb_matrix[x:x + scale, y + scale, :] = [0, 0, 0]

                elif val == 'door':
                    rgb_matrix[x:x + scale, y:y + scale, :] = [50, 50, 50]

                elif val == 'brick':
                    rgb_matrix[x:x + scale, y:y + scale, :] = [150, 100, 50]

                elif val == 'red_tile':
                    rgb_matrix[x:x + scale, y:y + scale, :] = [150, 50, 30]

                elif val == 'land':
                    rgb_matrix[x:x + scale, y:y + scale, :] = [0, 50, 0]

                elif val == 'water':
                    rgb_matrix[x:x + scale, y:y + scale, :] = [50, 50, 100]

                elif val == 'bridge':
                    rgb_matrix[x:x + scale, y:y + scale, :] = [100, 100, 100]

                elif val == 'target':
                    rgb_matrix[x:x + scale, y:y + scale, :] = [50, 90, 25]

                if (x_g, y_g) == (self.x, self.y):
                    rgb_matrix[x:x + scale, y:y + scale, :] = [200, 200, 200]

        return rgb_matrix

    def num_house(self):
        h = len(self.house_image[0])
        w = len(self.house_image)
        count = 0
        for x in range(0, GRID_DIM[0]):
            for y in range(0, GRID_DIM[1]):
                if x + w < GRID_DIM[0] and y + h < GRID_DIM[1]:
                    found = True
                    for i in range(0, w):
                        for j in range(0, h):
                            if self.house_image[i][j] is not None \
                                and self.house_image[i][j] != self.grid[
                                    (x + i, y + j)]:
                                found = False
                    if found:
                        count += 1
        return count

    def is_goal_state(self):
        return self.num_house() == 1


def main():
    env = ConstructBaseEnv()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(400):
            env.render()
            # print(observation)
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            print("reward : ", reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == "__main__":
    main()
