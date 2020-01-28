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

import sys
import os
import pygame
import numpy as np
import datetime
import random
import math
import operator
import gym
from gym import spaces, logger
import cv2
from pygame.locals import *
from pygame.color import THECOLORS
from matplotlib import pyplot as plt
GRID_DIM = (32, 32)

HOUSE_IMAGE = \
    [[None, 'red_tile', 'red_tile', 'red_tile', None],
     ['red_tile', 'red_tile', 'red_tile', 'red_tile', 'red_tile'],
     [None, 'brick', 'brick', 'brick', None],
     [None, 'brick', 'door', 'brick', None]]


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_rgb_matrix(rgb_matrix, m=15):
    rows = rgb_matrix.shape[0]
    cols = rgb_matrix.shape[1]

    # rgb_array = np.ones_like(rgb_matrix).astype('uint8') * 50
    rgb_array = np.ones((rows * m, cols * m, 3)) * 50
    # pygame.draw.rect(screen, pygame.Color(50, 50, 50), pygame.Rect(0 + x_offset, 0, 800, 600))

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
    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(
                GRID_DIM[0], GRID_DIM[1], 3))
        self.house_image = \
            [[None, 'red_tile', 'red_tile', 'red_tile', None],
             ['red_tile', 'red_tile', 'red_tile', 'red_tile', 'red_tile'],
             [None, 'brick', 'brick', 'brick', None],
             [None, 'brick', 'door', 'brick', None]]
        self.max_steps = 400
        self.num_envs = 1
        self.reset()

    def reset(self):
        self.grid = {}
        self.prev_num_house = 0
        for i in range(GRID_DIM[0]):
            for j in range(GRID_DIM[1]):
                self.grid[(i, j)] = 'empty'

        self.regions = np.zeros(GRID_DIM)

        self.steps = 0

        rows = GRID_DIM[0]
        cols = GRID_DIM[1]

        self.t = 0

        self.centroids = {}
        self.region_sizes = {}
        self.create_regions()

        smallest = min(
            self.region_sizes.items(),
            key=operator.itemgetter(1))[0]
        largest = max(self.region_sizes.items(), key=operator.itemgetter(1))[0]
        # print("size: ",smallest,largest)

        (self.x, self.y) = self.centroids[smallest]
        (self.target_x, self.target_y) = self.centroids[largest]

        # print("target: ",self.target_x,self.target_y)

        '''
        for i in range(-1,2):
            for j in range(-1,2):
                if (abs(i)==1 and abs(j)==1) or (i==0 and j==0):
                    if i+self.target_x>=0 and \
                    i+self.target_x<self.regions.shape[0] and \
                    j+self.target_y>=0 and \
                    j+self.target_y<self.regions.shape[1]:
                        self.grid[(i+self.target_x,j+self.target_y)]='target'
        '''
        self.reset_if_invalid()
        self.chosen_sites = []

        count = 0
        found = False
        while not found:
            self.chosen_sites = []
            found = True
            for i in range(3):
                if not self.choose_house_site():
                    found = False
            count += 1
            if count > 10 and not found:
                # print("reset needed")
                self.__init__()
                return self.get_rgb_matrix()

        random.shuffle(self.chosen_sites)

        for chosen_site in self.chosen_sites:
            x = chosen_site[0]
            y = chosen_site[1]
            self.grid[(x + 3, y + 1)] = 'door'
            self.grid[(x + 3, y + 2)] = 'door'
            self.grid[(x + 3, y + 3)] = 'door'

        return self.get_rgb_matrix()

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

    def create_regions(self):

        failed = False

        for i in range(self.regions.shape[0]):
            for j in range(self.regions.shape[1]):
                self.grid[(i, j)] = 'land'

        sides = ['l', 'r', 't', 'b']

        random.shuffle(sides)

        sides = sides[0:3]

        regions = [1, 2, 3]

        for i in range(3):
            if sides[i] == 'l':
                self.regions[0,
                             random.choice(range(self.regions.shape[1]))] = \
                    regions[i]
                # self.grid[   0,random.choice(range(self.regions.shape[1]))]=regions[i]

            if sides[i] == 'r':
                self.regions[self.regions.shape[0] - 1,
                             random.choice(range(self.regions.shape[1]))] = \
                    regions[i]
                # self.grid[   self.regions.shape[0]-1,random.choice(range(self.regions.shape[1]))]=regions[i]

            if sides[i] == 't':
                self.regions[random.choice(range(self.regions.shape[0])),
                             0] = \
                    regions[i]
                # self.grid[   random.choice(range(self.regions.shape[0])),0]=regions[i]

            if sides[i] == 'b':
                self.regions[random.choice(range(self.regions.shape[0])),
                             self.regions.shape[1] - 1] = \
                    regions[i]
                # self.grid[   random.choice(range(self.regions.shape[0])),self.regions.shape[1]-1]=regions[i]

        for count in range(30):

            locations = []
            for i in range(self.regions.shape[0]):
                for j in range(self.regions.shape[1]):
                    if self.regions[i, j] != 0:
                        locations.append((i, j))

            random.shuffle(locations)

            for location in locations:
                self.fill(location)

        for region in regions:
            i_points = []
            j_points = []
            count = 0
            for i in range(self.regions.shape[0]):
                for j in range(self.regions.shape[1]):
                    if self.regions[i, j] == region:
                        i_points.append(i)
                        j_points.append(j)
                        count += 1

            if len(i_points) != 0 and len(j_points) != 0:

                i_avg = int(sum(i_points) / len(i_points))
                j_avg = int(sum(j_points) / len(j_points))
                self.centroids[region] = (i_avg, j_avg)
                self.region_sizes[region] = count

            else:
                self.centroids[region] = (-1, -1)
                self.region_sizes[region] = 0

        for i in range(self.regions.shape[0]):
            for j in range(self.regions.shape[1]):
                self.paint_if_border((i, j))

        return failed

    def paint_if_border(self, location):
        (row, col) = location

        color = self.regions[row, col]

        border = False
        for i in range(max(0, row - 1),
                       min(self.regions.shape[0], row + 2)):
            for j in range(max(0, col - 1),
                           min(self.regions.shape[1], col + 2)):
                if self.regions[i, j] != color:
                    border = True
        if border:
            self.grid[(i, j)] = 'water'

    def fill(self, location):

        (row, col) = location

        color = self.regions[row, col]

        filled = False
        for i in range(max(0, row - 1),
                       min(self.regions.shape[0], row + 2)):
            for j in range(max(0, col - 1),
                           min(self.regions.shape[1], col + 2)):
                if self.regions[i, j] == 0:
                    self.regions[i, j] = color
                    # self.grid[i,j]=color
                    filled = True

        return filled

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

        if action == 0:
            if self.y + \
                    1 < cols and self.grid[(self.x, self.y + 1)] != 'water':
                self.y += 1

        if action == 1:
            if self.y - 1 >= 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.y -= 1

        if action == 2:
            if self.x - 1 >= 0 and self.grid[(self.x - 1, self.y)] != 'water':
                self.x -= 1

        if action == 3:
            if self.x + \
                    1 < rows and self.grid[(self.x + 1, self.y)] != 'water':
                self.x += 1

        if action == 4:
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'brick'

        if action == 5:
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'door'

        if action == 6:
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'red_tile'

        if action == 7:
            self.build_bridge()

        if action == 8:
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'land'

        if action == 9:
            pass

        # if action == 'reset':
        #     self.reset()
        #     terminated = True

        self.t += 1

        state = self.get_rgb_matrix()

        cur_num_house = self.num_house()
        rew = (cur_num_house - self.prev_num_house) * 10
        self.prev_num_house = cur_num_house

        # terminated = self.is_goal_state()
        self.steps += 1
        terminated = self.is_goal_state() or (self.steps >= self.max_steps)

        return state, rew, terminated, {}
        # return np.array([state]), \
        #     np.array([rew]), \
        #     np.array([terminated]), \
        #     {}

    def render(self):
        rgb_matrix = self.get_rgb_matrix()
        # cv2.imshow('simulate', cv2.resize(draw_rgb_matrix(rgb_matrix).astype('uint8')[:,:,::-1],
        #                                   (256, 256), interpolation=cv2.INTER_LANCZOS4))
        cv2.imshow('simulate',
                   draw_rgb_matrix(rgb_matrix).astype('uint8')[:, :, ::-1])
        cv2.waitKey(10)
        # plt.imshow(cv2.resize(draw_rgb_matrix(rgb_matrix), (256, 256)))
        # plt.pause(0.01)

    def build_bridge(self):
        row = self.x
        col = self.y
        for i in range(max(0, row - 1), min(self.regions.shape[0], row + 2)):
            for j in range(max(0, col - 1),
                           min(self.regions.shape[1], col + 2)):
                if self.grid[(i, j)] == 'water':
                    self.grid[(i, j)] = 'bridge'

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
        return self.num_house() == 3


def main():
    env = ConstructBaseEnv()
    for i_episode in range(20):
        observations = env.reset()
        while True:
            env.render()
            # print(observation)
            action = env.action_space.sample()

            observations, rewards, dones, infos = env.step(action)
            print("reward : ", rewards[0])
            done = dones.any() if isinstance(dones, np.ndarray) else dones
            if done:
                print("Terminated")
                break


if __name__ == "__main__":
    main()
