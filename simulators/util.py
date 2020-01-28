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
import math
import numpy as np
import cv2

ACTIONS = ['NOOP', 'FIRE','UP','RIGHT','LEFT','DOWN','UPRIGHT','UPLEFT','DOWNRIGHT','DOWNLEFT','UPFIRE','RIGHTFIRE','LEFTFIRE','DOWNFIRE','UPRIGHTFIRE','UPLEFTFIRE','DOWNRIGHTFIRE','DOWNLEFTFIRE']

# this list is mostly needed to list the games in the same order everywhere
GAMES = ['spaceinvaders', 'qbert', 'mspacman', 'pinball','revenge']

# pretty titles for plots/tables
TITLES = {'spaceinvaders': 'Space Invaders',
          'qbert': 'Q*bert',
          'mspacman':'Ms. Pacman',
          'pinball':'Video Pinball',
          'revenge':'Montezumas\'s Revenge'
         }

def preprocess(state, resize_shape=(84,84,3)):
    # Resize state
    #state = cv2.resize(state, resize_shape)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
    #state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)

    '''
    if len(state.shape) == 3:
        if state.shape[2] == 3:
    '''

    # Check type is compatible
    if state.dtype != np.float32:
        state = state.astype(np.float32)

    '''
    # normalize
    if state.max() > 1:
        state *= 1. / 255.

    return state.reshape(-1, 84, 84)
    '''
    return state


def get_action_name(action_code):
  assert 0 <= action_code < len(ACTIONS), "%d is not the valid action index." % action_code
  return ACTIONS[action_code]

def get_action_code(action_name):
    assert action_name in ACTIONS, "%s is not the valid action name." % action_name
    return ACTIONS.index(action_name)

def env2game(name):
    ENVS = {'SpaceInvaders-v3': 'spaceinvaders', 
             'MsPacman-v3':'mspacman', 
             'VideoPinball-v3':'pinball',
             'MontezumaRevenge-v3':'revenge',
             'Qbert-v3':'qbert'
            }
    return ENVS[name]

def game2env(name):
    GAMES = {'spaceinvaders':'SpaceInvaders-v3', 
             'mspacman':'MsPacman-v3', 
             'pinball':'VideoPinball-v3',
             'revenge':'MontezumaRevenge-v3',
             'qbert':'Qbert-v3'
            }
    return GAMES[name]
