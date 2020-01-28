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
import sys,os
import pygame
import numpy as np
import datetime
import random

from pygame.locals import *
from pygame.color import THECOLORS

GRID_DIM = (20,10)

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
                elif (event.key==K_1):            
                    return 1           
                elif (event.key==K_2):                          
                    return 2
                elif (event.key==K_3):                          
                    return 3
                elif (event.key==K_4):                          
                    return 4
                elif (event.key==K_5):                          
                    return 5
                elif(event.key==K_SPACE):
                    return 'space'
                elif(event.key==K_f):
                    return 'f'
                elif(event.key==K_d):
                    return 'd'
                elif(event.key==K_LEFT):
                    return 'left'
                elif(event.key==K_RIGHT):
                    return 'right'
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
        self.name='bridge'
        self.grid=np.zeros(GRID_DIM)
        #self.grid=np.random.randint(5,size=GRID_DIM)

        rows=self.grid.shape[0]
        cols=self.grid.shape[1]

        self.t=0

        self.mode='cars'
        #self.mode='no_cars'
        
        self.height=np.random.randint(3,8)
        l=self.height
        r=self.height

        stop=0

        for x in range(0, rows):
            for y in range(0, cols):
                if y==0:
                    self.grid[x,y]=2
                if y<l:
                    self.grid[x,y]=1

            if random.random()<0.5 and l>0 and x>3:
                l-=1
                stop=x
                

        for x in reversed(list(range(stop,rows))):
            for y in range(0, cols):
                if y<r:
                    self.grid[x,y]=1
            if random.random()<0.5 and r>0:
                r-=1

        self.desp_pos=4
        self.grid[self.desp_pos,cols-1]=5

                    
    def step(self,action):
        rows=self.grid.shape[0]
        cols=self.grid.shape[1]
        if action=='drop_block':
            self.grid[self.desp_pos,cols-2]=3

        if action=='drop_strut':
            self.grid[self.desp_pos-1:self.desp_pos+2,cols-2]=4
            self.grid[self.desp_pos-2,cols-2]=6
            self.grid[self.desp_pos+2,cols-2]=6


        if action=='left':
            if self.desp_pos>0:
                self.grid[self.desp_pos,cols-1]=0
                self.desp_pos-=1
                self.grid[self.desp_pos,cols-1]=5

        if action=='right':
            if self.desp_pos<rows-1:
                self.grid[self.desp_pos,cols-1]=0
                self.desp_pos+=1
                self.grid[self.desp_pos,cols-1]=5



        for x in range(0, rows):
            for y in range(0, cols):
                if self.grid[x,y]==3:
                    if y>0 and self.grid[x,y-1]==0:
                        self.grid[x,y-1]=3
                        self.grid[x,y]=0

                if self.grid[x,y]==6:
                    if y>0:
                        if self.grid[x,y-1]==0:
                            self.grid[x,y-1]=6
                            self.grid[x,y]=0
                        elif self.grid[x,y-1]==6:
                            self.grid[x,y]=0

                
        for x in range(0, rows):
            for y in range(0, cols):
                if self.grid[x,y]==4:
                    supported=True 

                    c=x
                    while c>=0 and self.grid[c,y]==4:
                        c-=1

                    if self.grid[c,y]!=6:
                        supported=False
                        
                    c=x
                    while c<rows and self.grid[c,y]==4:
                        c+=1

                    if self.grid[c,y]!=6:
                        supported=False

                    if not supported:
                        if y>0 and self.grid[x,y-1]==0:
                            self.grid[x,y-1]=self.grid[x,y]
                            self.grid[x,y]=0
        if self.mode=='cars':
            h=self.height
            if self.t%10==0:
                for x in reversed(list(range(0, rows))):
                    if self.grid[x,h]==8:
                        if x==rows-1:
                            print("WIN")
                            self.__init__()
                            break
                        if self.grid[x,h-1]==0:
                            print("GAME OVER")
                            self.__init__()
                            break
                       
                        if self.grid[x+1,h]==0:
                            self.grid[x+1,h]=self.grid[x,h]
                            self.grid[x,h]=0
                        else:
                            print("GAME OVER")
                            self.__init__()
                            break


            if self.t%100==0 and self.t>0:
                self.grid[0,h]=8

        self.t+=1 



    def draw(self,screen):

        rows=self.grid.shape[0]
        cols=self.grid.shape[1]

        pygame.draw.rect(screen,pygame.Color(50,50,50,50), pygame.Rect(0,0,640,480))

        for x in range(0, rows):
            for y in range(0, cols):
                color=THECOLORS['black']
                if self.grid[x,y]==1:
                    color=THECOLORS['green']
                elif self.grid[x,y]==2:
                    color=THECOLORS['blue']
                elif self.grid[x,y]==3:
                    color=pygame.Color(200,50,50,100)
                elif self.grid[x,y]==4:
                    color=pygame.Color(150,150,150,150)
                elif self.grid[x,y]==5:
                    color=THECOLORS['orange']
                elif self.grid[x,y]==6:
                    color=pygame.Color(100,100,100,100)
                elif self.grid[x,y]==8:
                    color=pygame.Color(100,200,200,100)


                pygame.draw.rect(screen,color, pygame.Rect(x*20-1,400-y*20-1,18,18))


def main():

    world=World()
    
    pygame.init()

    window_size = (640, 480)

    window=Window(window_size)
    screen=window.surface
    window.update_title(__file__)

    myclock=pygame.time.Clock()
        
    framerate_limit=20

    t=0.0
    finished=False
    
    while not finished:
        
        window.surface.fill(THECOLORS["black"])
        dt=float(myclock.tick(framerate_limit) * 1e-3)
        
        command=window.get_user_input()


        if command=='reset':
            world=World()
            world.draw(screen)
        
        elif (command=='quit'):
            finished = True
            
        elif (command!=None):
            pass

        if command=='space' or command==None:
            action='no_action'
            world.step(action)

        if command=='d':
            action='drop_block'
            world.step(action)

        if command=='f':
            action='drop_strut'
            world.step(action)

        if command=='left':
            action='left'
            world.step(action)

        if command=='right':
            action='right'
            world.step(action)
        world.draw(screen) 
        
        t+=dt
        pygame.display.flip()

if __name__=="__main__":
    main()
