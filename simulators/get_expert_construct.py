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
import sys, os
import pygame
import numpy as np
import datetime
import random
import math
import operator

from pygame.locals import *
from pygame.color import THECOLORS
from construct_base_gym import draw_rgb_matrix
import cv2
from copy import deepcopy

GRID_DIM = (32, 32)
HOUSE_IMAGE = [[None, 'red_tile', 'red_tile', 'red_tile', None], \
               ['red_tile', 'red_tile', 'red_tile', 'red_tile', 'red_tile'], \
               [None, 'brick', 'brick', 'brick', None], \
               [None, 'brick', 'door', 'brick', None]]


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class Window:
    def __init__(self, screen_dim):
        self.width_px = screen_dim[0]
        self.height_px = screen_dim[1]

        self.surface = pygame.display.set_mode(screen_dim)
        self.black_update()

    def update_title(self, title):
        pygame.display.set_caption(title)
        self.title = title

    def black_update(self):
        self.surface.fill(THECOLORS["black"])
        pygame.display.flip()

    def get_user_input(self):

        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                return 'quit'
            elif (event.type == pygame.KEYDOWN):
                if (event.key == K_ESCAPE):
                    return 'quit'
                elif (event.key == K_1):
                    return '1'
                elif (event.key == K_2):
                    return '2'
                elif (event.key == K_3):
                    return '3'
                elif (event.key == K_4):
                    return 'bridge'
                elif (event.key == K_5):
                    return '5'
                elif (event.key == K_UP):
                    return 'up'
                elif (event.key == K_DOWN):
                    return 'down'
                elif (event.key == K_LEFT):
                    return 'left'
                elif (event.key == K_RIGHT):
                    return 'right'
                elif (event.key == K_m):
                    return "m"
                elif (event.key == K_RETURN):
                    return "reset"
                else:
                    return "Nothing set up for this key."

            elif (event.type == pygame.KEYUP):
                pass

            elif (event.type == pygame.MOUSEBUTTONDOWN):
                pass

            elif (event.type == pygame.MOUSEBUTTONUP):
                pass


class World:
    def __init__(self):
        self.grid = {}
        for i in range(GRID_DIM[0]):
            for j in range(GRID_DIM[1]):
                self.grid[(i, j)] = 'empty'

        self.regions = np.zeros(GRID_DIM)

        rows = GRID_DIM[0]
        cols = GRID_DIM[1]

        self.t = 0

        self.centroids = {}
        self.region_sizes = {}
        self.create_regions()

        smallest = min(self.region_sizes.items(), key=operator.itemgetter(1))[0]
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
                print("reset needed")
                self.__init__()
                return

        random.shuffle(self.chosen_sites)

        for chosen_site in self.chosen_sites:
            x = chosen_site[0]
            y = chosen_site[1]
            self.grid[(x + 3, y + 1)] = 'door'
            self.grid[(x + 3, y + 2)] = 'door'
            self.grid[(x + 3, y + 3)] = 'door'

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

        for i in range(max(0, x - distance), min(GRID_DIM[0], x + distance + 1)):
            for j in range(max(0, y - distance), min(GRID_DIM[1], y + distance + 1)):
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
                self.regions[0, random.choice(range(self.regions.shape[1]))] = regions[i]
                # self.grid[   0,random.choice(range(self.regions.shape[1]))]=regions[i]

            if sides[i] == 'r':
                self.regions[self.regions.shape[0] - 1, random.choice(range(self.regions.shape[1]))] = regions[i]
                # self.grid[   self.regions.shape[0]-1,random.choice(range(self.regions.shape[1]))]=regions[i]

            if sides[i] == 't':
                self.regions[random.choice(range(self.regions.shape[0])), 0] = regions[i]
                # self.grid[   random.choice(range(self.regions.shape[0])),0]=regions[i]

            if sides[i] == 'b':
                self.regions[random.choice(range(self.regions.shape[0])), self.regions.shape[1] - 1] = regions[i]
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
        for i in range(max(0, row - 1), min(self.regions.shape[0], row + 2)):
            for j in range(max(0, col - 1), min(self.regions.shape[1], col + 2)):
                if self.regions[i, j] != color:
                    border = True
        if border:
            self.grid[(i, j)] = 'water'

    def fill(self, location):

        (row, col) = location

        color = self.regions[row, col]

        filled = False
        for i in range(max(0, row - 1), min(self.regions.shape[0], row + 2)):
            for j in range(max(0, col - 1), min(self.regions.shape[1], col + 2)):
                if self.regions[i, j] == 0:
                    self.regions[i, j] = color
                    # self.grid[i,j]=color
                    filled = True

        return filled

    def step(self, action):
        terminated = False

        rows = GRID_DIM[0]
        cols = GRID_DIM[1]

        if action == 'up':
            if self.y + 1 < cols and self.grid[(self.x, self.y + 1)] != 'water':
                self.y += 1

        if action == 'down':
            if self.y - 1 >= 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.y -= 1

        if action == 'left':
            if self.x - 1 >= 0 and self.grid[(self.x - 1, self.y)] != 'water':
                self.x -= 1

        if action == 'right':
            if self.x + 1 < rows and self.grid[(self.x + 1, self.y)] != 'water':
                self.x += 1

        if action == '1' or action == 'brick':
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'brick'

        if action == '2' or action == 'door':
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'door'

        if action == '3' or action == 'red_tile':
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'red_tile'

        if action == '4' or action == 'land':
            if self.y - 1 > 0 and self.grid[(self.x, self.y - 1)] != 'water':
                self.grid[(self.x, self.y - 1)] = 'land'

        if action == 'bridge':
            self.build_bridge()

        if action == 'reset':
            self.__init__()
            terminated = True

        self.t += 1

        return terminated

    def build_bridge(self):

        row = self.x
        col = self.y

        for i in range(max(0, row - 1), min(self.regions.shape[0], row + 2)):
            for j in range(max(0, col - 1), min(self.regions.shape[1], col + 2)):
                if self.grid[(i, j)] == 'water':
                    self.grid[(i, j)] = 'bridge'

    def match(self, x, y, c):
        if self.grid[x, y] == c or self.grid[x, y] - 4 == c or self.grid[x, y] == c - 4:
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


class AutoConstruct:
    def __init__(self, world, padding=None):
        self.world = world
        self.padding = padding

        self.foundation_image = [[None, None, None, None, None], \
                                 [None, None, None, None, None], \
                                 [None, None, None, None, None], \
                                 [None, 'door', 'door', 'door', None]]

        self.mask = world.grid.copy()

        self.subtasks = ['construct_house', 'construct_house', 'construct_house', 'move_to_site_centroid']
        # self.subtasks=['construct_house']

        if False and padding == 'random_moves':

            for i in range(4):
                self.subtasks = ['random_move'] + self.subtasks + ['random_move']

        self.complete = True
        self.prev_rew = 0
        self.generate_masks()

    def generate_masks(self):
        image = HOUSE_IMAGE

        h = len(image[0])
        w = len(image)

        self.masks = []
        current_mask = self.world.grid.copy()

        for chosen_site in self.world.chosen_sites:
            x = chosen_site[0]
            y = chosen_site[1]
            if x + w < GRID_DIM[0] and y + h < GRID_DIM[1]:
                for i in range(0, w):
                    for j in range(0, h):
                        if image[i][j] != None:
                            current_mask[(x + i, y + j)] = image[i][j]

            self.masks.insert(0, current_mask.copy())

    def auto_step(self):

        annotation = 'none'
        action = 'none'

        if self.complete:
            if len(self.subtasks) > 0:
                self.subtask = self.subtasks.pop()
                self.complete = False
                self.pstep = 0
            else:
                action = 'reset'
                self.complete = False

        else:

            if self.subtask == 'move_to_site_centroid':
                action = self.subtask_move_to_site_centroid()

            if self.subtask == 'construct_house':
                action, annotation = self.subtask_construct_house()

            if self.subtask == 'random_move':
                action = self.subtask_random_move()

            if self.complete:
                annotation = "completed: " + str(self.subtask)
                action = 'none'
                terminal = True

                if self.is_goal_state():
                    reward = 1.0
                else:
                    reward = 0.0

                return action, annotation, reward, terminal

        terminal = self.world.step(action)

        reward = self.num_house() - self.prev_rew
        self.prev_rew = self.num_house()

        if action == 'reset':
            self.__init__(self.world, padding=self.padding)

        # print(action)
        return action, annotation, reward, terminal

    def subtask_random_move(self):

        if self.pstep == 0:
            self.ptarget = random.choice(self.world.centroids.values())
            self.pstep += 1

        if self.pstep >= 1:
            (x, y) = self.ptarget
            action = self.calc_action(x, y, 'move')
            self.pstep += 1

        if action == None:
            self.complete = True

        return action

    def subtask_move_to_site_centroid(self):

        action = self.calc_action(self.world.target_x, self.world.target_y, 'move')

        if action == None:
            self.complete = True

        return action

    def subtask_construct_house(self):

        action = 'none'
        annotation = 'none'

        if self.pstep == 0:
            '''
            self.cur_site=self.world.chosen_sites.pop()
            self.mask=self.world.grid.copy()
            (x,y)=self.cur_site
            self.dests=[]
            self.first_construct=False

            house_dests=self.mask_image(x,y,HOUSE_IMAGE)
            random.shuffle(house_dests)

            self.dests=house_dests+self.dests
            '''

            self.dests = []
            self.first_construct = False
            mask = self.masks.pop()
            self.dests = self.get_dests(mask) + self.dests

            self.pstep += 1

        if self.pstep == 1:

            if len(self.dests) > 0:
                self.dest = self.dests.pop()
                self.pstep += 1
            else:
                self.complete = True
                return action, annotation

        if self.pstep >= 2:
            action = self.calc_action(self.dest[0], self.dest[1], self.dest[2])
            if action == None:
                self.pstep = 1
                self.subtask_construct_house()

        if action == 'door' and not self.first_construct:
            annotation = 'begin: construct_house'
            self.first_construct = True

        return action, annotation

    def get_dests(self, mask):
        changes = []
        for i in range(GRID_DIM[0]):
            for j in range(GRID_DIM[1]):
                if mask[(i, j)] != self.world.grid[(i, j)] and mask[(i, j)] != 'water' and mask[(i, j)] != 'door':
                    changes.append((i, j, mask[(i, j)]))
        random.shuffle(changes)
        return changes

    def mask_image(self, x, y, image, mask):

        h = len(image[0])
        w = len(image)

        changes = []

        '''
        if x+w<GRID_DIM[0] and y+h<GRID_DIM[1]:
            for i in range(0,w):
                for j in range(0,h):
                    if image[i][j]!=None:
                        self.mask[(x+i,y+j)]=image[i][j]
                        changes.append((x+i,y+j,image[i][j]))
        '''

        if x + w < GRID_DIM[0] and y + h < GRID_DIM[1]:
            for i in range(0, w):
                for j in range(0, h):
                    if image[i][j] != None:
                        self.mask[(x + i, y + j)] = image[i][j]
                        changes.append((x + i, y + j, image[i][j]))
        return changes

    def calc_action(self, dx, dy, color):
        action = 'none'
        dy = dy + 1
        if abs(dx - self.world.x) > abs(dy - self.world.y):
            if dx < self.world.x:
                if self.world.x - 1 >= 0 and self.world.grid[(self.world.x - 1, self.world.y)] != 'water':
                    return 'left'
                else:
                    return 'bridge'
            elif dx > self.world.x:
                if self.world.x + 1 < GRID_DIM[0] and self.world.grid[(self.world.x + 1, self.world.y)] != 'water':
                    return 'right'
                else:
                    return 'bridge'
        else:
            if dy < self.world.y:
                if self.world.y - 1 >= 0 and self.world.grid[(self.world.x, self.world.y - 1)] != 'water':
                    return 'down'
                else:
                    return 'bridge'

            elif dy > self.world.y:
                if self.world.y + 1 < GRID_DIM[1] and self.world.grid[(self.world.x, self.world.y + 1)] != 'water':
                    return 'up'
                else:
                    return 'bridge'

        if color != 'move' and self.world.grid[(self.world.x, self.world.y - 1)] != color:
            return color
        else:
            return None

    def is_goal_state(self):
        return self.num_house() == 3

    def num_house(self):
        h = len(HOUSE_IMAGE[0])
        w = len(HOUSE_IMAGE)
        count = 0
        for x in range(0, GRID_DIM[0]):
            for y in range(0, GRID_DIM[1]):
                if x + w < GRID_DIM[0] and y + h < GRID_DIM[1]:
                    found = True
                    for i in range(0, w):
                        for j in range(0, h):
                            if HOUSE_IMAGE[i][j] != None and HOUSE_IMAGE[i][j] != self.world.grid[(x + i, y + j)]:
                                found = False
                    if found:
                        count += 1
        return count


class Construct:
    def __init__(self, auto_padding=None):
        self.name = 'construct'
        self.world = World()
        self.auto_construct = AutoConstruct(self.world, padding=auto_padding)

    def reset(self):
        self.__init__()

    def step(self, action):
        self.world.step(action)

    def auto_step(self):
        state0 = self.world.get_rgb_matrix()
        action, annotation, reward, terminated = self.auto_construct.auto_step()
        state1 = self.world.get_rgb_matrix()
        return state0, state1, action, annotation, reward, terminated, self.world.t

    def random_step(self):
        actions = ['up', 'down', 'left', 'right', '1', '2', '3', '4']
        action = random.choice(actions)
        state0 = self.world.get_rgb_matrix()
        reward, terminated = self.world.step(action)
        annotation = None
        state1 = self.world.get_rgb_matrix()
        return state0, state1, action, annotation, reward, terminated, self.world.t

    def set_auto_padding(self, padding):
        self.auto_construct.padding = padding

    def draw(self, screen):
        rgb_matrix = self.world.get_rgb_matrix()
        draw_rgb_matrix(screen, rgb_matrix)


def main():
    import imageio
    vid_writer = imageio.get_writer('expertVideo.mp4', fps=20)

    pygame.init()
    mode = 'auto'
    finished = False
    render = True

    expert_data = {'obs': [], 'acs': [], 'rews': [], 'done': [], 'next_obs': [], 'ep_rets': []}

    savefile = 'expertData.npz'

    for ep in range(100):
        print('Episode : ', ep)
        ep_ret = 0
        world = World()
        auto_construct = AutoConstruct(world, 'random_moves')
        sub_expert_data = {'obs': [], 'acs': [], 'rews': [], 'done': [], 'next_obs': []}
        obs = world.get_rgb_matrix()
        while not finished:
            sub_expert_data['obs'].append(deepcopy(obs))
            action, annotation, reward, terminated = auto_construct.auto_step()
            sub_expert_data['acs'].append(0)
            sub_expert_data['rews'].append(reward)

            next_obs = world.get_rgb_matrix()
            sub_expert_data['next_obs'].append(deepcopy(next_obs))

            obs = deepcopy(next_obs)

            if render:
                img = draw_rgb_matrix(next_obs).astype('uint8')
                vid_writer.append_data(img)
                cv2.imshow('simulate', img[:, :, ::-1])
                cv2.waitKey(10)

            terminated = auto_construct.is_goal_state()

            sub_expert_data['done'].append(terminated)

            ep_ret += reward

            if terminated:
                break

        for key in sub_expert_data.keys():
            expert_data[key].append(sub_expert_data[key])

        expert_data['ep_rets'].append(ep_ret)

    np.savez(savefile, **expert_data)
    vid_writer.close()


if __name__ == "__main__":
    main()
