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
import util
import pygame
from pygame.locals import *
from pygame.color import THECOLORS
import os
import random
import cv2
import numpy as np

DATA_DIR ="/home/everett/datasets/atari_v1"

class Window:
    def __init__(self):
        self.width_px = 160
        self.height_px = 210

        self.surface = pygame.display.set_mode((210,160))
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
                elif (event.key==K_1):            
                    return '1'           
                elif (event.key==K_2):                          
                    return '2'
                elif (event.key==K_3):                          
                    return '3'
                elif (event.key==K_4):                          
                    return '4'
                elif (event.key==K_5):                          
                    return '5'
                elif(event.key==K_UP):
                    return 'up'
                elif(event.key==K_DOWN):
                    return 'down'
                elif(event.key==K_LEFT):
                    return 'left'
                elif(event.key==K_RIGHT):
                    return 'right'
                elif(event.key==K_m):
                    return "m"
                elif(event.key==K_RETURN):
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

        self.eps_names=os.listdir(DATA_DIR+'/screens/mspacman/')

        self.load_ep(random.choice(self.eps_names))
        
    def load_ep(self,ep_name):

        print("LOADING EP: ",ep_name)
        self.data = []

        ep_dir=DATA_DIR+'/screens/mspacman/'+ep_name

        file_names= os.listdir(ep_dir)
        i=500
        found=True
        while found:
            try:
                image=cv2.imread(ep_dir+'/'+str(i)+'.png')
                if image.shape[0]!=0:
                    state = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    state=np.rot90(state,axes=(0,1))
                    state=np.flip(state,0)

                    #state=state.transpose()
                    self.data.append(state)
                    i+=4
            except:
                found=False

        self.t=0

    def step(self):
        reward=0.0
        terminated=False
        action="null"
        annotation="none"
        self.t+=1

        if self.t==len(self.data):
            annotation='reset'
            self.load_ep(random.choice(self.eps_names))


        return action,annotation,reward,terminated

    def get_rgb_matrix(self):
        return self.data[self.t]


class MsPacman:
    def __init__(self):
        self.name='ms_pacman'
        self.world=World()

    def step(self,action):
        pass

    def auto_step(self):
        state0=self.world.get_rgb_matrix()
        action,annotation,reward,terminated=self.world.step()
        state1=self.world.get_rgb_matrix()
        return state0,state1,action,annotation,reward,terminated,self.world.t

def main():

    world=World()
    
    pygame.init()


    window=Window()
    screen=window.surface
    window.update_title("Ms. Pacman")

    myclock=pygame.time.Clock()
        
    framerate_limit=40

    t = 0.0
    finished=False
    mode='user'
    
    while not finished:
        
        window.surface.fill(THECOLORS["black"])
        dt=float(myclock.tick(framerate_limit) * 1e-3)
        
        command=window.get_user_input()

        if command:
            pass

        if command=='reset':
            world=World()
            #auto_paint=AutoPaint(world)
        
        elif (command=='quit'):
            finished=True
            
        elif (command!=None):
            pass

        if command=='m':
            if mode=='user':
                mode='auto'
            elif mode=='auto':
                mode='user'
        
        if mode=='auto':
            world.step()

        if     command=='up'\
            or command=='down'\
            or command=='left'\
            or command=='right'\
            or command=='1'\
            or command=='2'\
            or command=='3'\
            or command=='4':
                frame=world.step(command)
                print(type(frame))
                print(len(frame))

        #s0=world.get_rgb_matrix()
        #draw_rgb_matrix(screen,s0) 
        
        t+=dt
        pygame.display.flip()



if __name__=="__main__":
    main()
