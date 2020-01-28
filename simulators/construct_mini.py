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
import math
import operator
from copy import deepcopy


from pygame.locals import *
from pygame.color import THECOLORS

GRID_DIM = (8,8)
HOUSE_IMAGE= [[None,'red_tile','red_tile','red_tile',None],\
        ['red_tile','red_tile','red_tile','red_tile','red_tile'],\
        [None,'brick','brick','brick',None],\
        [None,'brick','door','brick',None]]

FOUNDATION_IMAGE= [[None,None,None,None,None],\
                [None,None,None,None,None],\
                [None,None,None,None,None],\
                [None,'door','door','door',None]]

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def draw_rgb_matrix(screen,rgb_matrix,x_offset=0):

    rows=rgb_matrix.shape[0]
    cols=rgb_matrix.shape[1]

    pygame.draw.rect(screen,pygame.Color(50,50,50), pygame.Rect(0+x_offset,0,800,600))

    for x in range(0, rows):
        for y in range(0, cols):
            r=int(rgb_matrix[y,x,0])
            g=int(rgb_matrix[y,x,1])
            b=int(rgb_matrix[y,x,2])
            if r<0:
                r=0
            elif r>255:
                r=255
            if b<0:
                b=0
            elif b>255:
                b=255
            if g<0:
                g=0
            elif g>255:
                g=255
            color=pygame.Color(r,g,b)
                
            #if val==4 or val==5 or val==6 or val==7:
            #    pygame.draw.rect(screen,pygame.Color(200,200,200), pygame.Rect(x*20,600-y*20,20,20))

            border=0
            pygame.draw.rect(screen,color, pygame.Rect(x*10+border+x_offset,y*10+border,10-2*border,10-2*border))

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
                    return '1'           
                elif (event.key==K_2):                          
                    return '2'
                elif (event.key==K_3):                          
                    return '3'
                elif (event.key==K_4):                          
                    return 'bridge'
                elif (event.key==K_5):                          
                    return '5'
                elif(event.key==K_UP):
                    return 'left'
                elif(event.key==K_DOWN):
                    return 'right'
                elif(event.key==K_LEFT):
                    return 'down'
                elif(event.key==K_RIGHT):
                    return 'up'
                elif(event.key==K_m):
                    return "m"
                elif(event.key==K_n):
                    return "count"
                elif(event.key==K_v):
                    return "view_subgoals"
                elif(event.key==K_g):
                    return "subgoal"
                elif(event.key==K_l):
                    return "load"
                elif(event.key==K_s):
                    return "save"
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
        self.grid={}
        for i in range(GRID_DIM[0]):
            for j in range(GRID_DIM[1]):
                self.grid[(i,j)]='empty'
                
        self.regions=np.zeros(GRID_DIM)

        rows=GRID_DIM[0]
        cols=GRID_DIM[1]

        self.t=0
        self.is_subgoal_state_indicator=False
        self.view_subgoals=False

        '''
        self.centroids={}
        self.region_sizes={}
        self.create_regions()
        '''
        for i in range(self.regions.shape[0]):
            for j in range(self.regions.shape[1]):
                self.grid[(i,j)]='land'
        self.checkpoints=[]
        
        '''
        smallest=min(self.region_sizes.items(), key=operator.itemgetter(1))[0]
        largest=max(self.region_sizes.items(), key=operator.itemgetter(1))[0]
        '''
        #print("size: ",smallest,largest)

        '''
        (self.x,self.y)=self.centroids[smallest] 
        (self.target_x,self.target_y)=self.centroids[largest]
        '''
        self.x=random.randint(0,rows-1)
        self.y=random.randint(0,cols-1)


        #print("target: ",self.target_x,self.target_y)

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


        #self.reset_if_invalid()
        site=(random.randint(1,2),random.randint(1,2))
        self.chosen_sites=[site]

        '''
        count=0
        found=False
        while not found:
            self.chosen_sites=[]
            found=True
            for i in range(3):
                if not self.choose_house_site():
                    found=False
            count+=1
            if count>10 and not found:
                print("reset needed")
                self.__init__()
                return

        random.shuffle(self.chosen_sites)
        '''

         
        self.goal_mask=self.grid.copy()
        for chosen_site in self.chosen_sites:
            x=chosen_site[0]
            y=chosen_site[1] 
            self.apply_mask(x,y,FOUNDATION_IMAGE,self.goal_mask)
            self.checkpoints.insert(0,self.goal_mask.copy())
            self.apply_mask(x,y,HOUSE_IMAGE,self.goal_mask)
            self.checkpoints.insert(0,self.goal_mask.copy())

    def export_state(self):
        return (self.grid,self.checkpoints,self.goal_mask)

    def load_state(grid,checkpoints,goal_mask):
        self.grid=grid.copy()
        self.checkpoints=list(checkpoints)
        self.goal_mask=goal_mask.copy()


    
    def apply_mask(self,x,y,image,mask):

        h=len(image[0])
        w=len(image)

        if x+w<GRID_DIM[0] and y+h<GRID_DIM[1]:
            for i in range(0,w):
                for j in range(0,h):
                    if image[i][j]!=None:
                        mask[(x+i,y+j)]=image[i][j]

    def choose_house_site(self):
        bad_site=True
        h=len(HOUSE_IMAGE[0])
        w=len(HOUSE_IMAGE)
        x=0
        y=0
        count=0
        while bad_site:
            x=self.target_x+random.randint(-10,10)
            y=self.target_y+random.randint(-10,10)
            
            bad_site=self.near_water_or_site(x,y,4)
            bad_site=bad_site or x+w>=GRID_DIM[0] or y+h>=GRID_DIM[1]
            count+=1

            if count>100:
                return False

        #self.world.grid[(x,y)]='red_tile'
        self.chosen_sites.append((x,y))

        return True

    def near_water_or_site(self,x,y,distance):

        if x<3 or x>=GRID_DIM[0]-3:
            return True
        if y<3 or y>=GRID_DIM[1]-3:
            return True


        for i in range(max(0,x-distance),min(GRID_DIM[0],x+distance+1)):
            for j in range(max(0,y-distance),min(GRID_DIM[1],y+distance+1)):
                if self.grid[(i,j)]=='water':
                    return True
                for (site_x,site_y) in self.chosen_sites:
                    if site_x==i and site_y==j:
                        return True

    def reset_if_invalid(self):

        reset=False
        for centroid in enumerate(self.centroids.items()):
            if centroid[0]==-1:
                reset=True

        if reset:
            print("reset")
            self.__init__()

    '''
    def create_regions(self):

        failed=False

        for i in range(self.regions.shape[0]):
            for j in range(self.regions.shape[1]):
                self.grid[(i,j)]='land'

        sides=['l','r','t','b']
        
        random.shuffle(sides)

        sides=sides[0:3]

        regions=[1,2,3]

        for i in range(3):
            if sides[i]=='l':
                self.regions[0,random.choice(range(self.regions.shape[1]))]=regions[i]
                #self.grid[   0,random.choice(range(self.regions.shape[1]))]=regions[i]

            if sides[i]=='r':
                self.regions[self.regions.shape[0]-1,random.choice(range(self.regions.shape[1]))]=regions[i]
                #self.grid[   self.regions.shape[0]-1,random.choice(range(self.regions.shape[1]))]=regions[i]

            if sides[i]=='t':
                self.regions[random.choice(range(self.regions.shape[0])),0]=regions[i]
                #self.grid[   random.choice(range(self.regions.shape[0])),0]=regions[i]

            if sides[i]=='b':
                self.regions[random.choice(range(self.regions.shape[0])),self.regions.shape[1]-1]=regions[i]
                #self.grid[   random.choice(range(self.regions.shape[0])),self.regions.shape[1]-1]=regions[i]


        for count in range(30):

            locations=[]
            for i in range(self.regions.shape[0]):
                for j in range(self.regions.shape[1]):
                    if self.regions[i,j]!=0:
                        locations.append((i,j))

            random.shuffle(locations)

            for location in locations:
                self.fill(location)
        
        for region in regions:
            i_points=[]
            j_points=[]
            count=0
            for i in range(self.regions.shape[0]):
                for j in range(self.regions.shape[1]):
                    if self.regions[i,j]==region:
                        i_points.append(i)
                        j_points.append(j)
                        count+=1


            if len(i_points)!=0 and len(j_points)!=0:

                i_avg=int(sum(i_points)/len(i_points))
                j_avg=int(sum(j_points)/len(j_points))
                self.centroids[region]=(i_avg,j_avg)
                self.region_sizes[region]=count

            else:
                self.centroids[region]=(-1,-1)
                self.region_sizes[region]=0


        for i in range(self.regions.shape[0]):
            for j in range(self.regions.shape[1]):
                self.paint_if_border((i,j))

        return failed



    def paint_if_border(self,location):
        (row,col)=location

        color=self.regions[row,col]

        border=False
        for i in range(max(0,row-1),min(self.regions.shape[0],row+2)):
            for j in range(max(0,col-1),min(self.regions.shape[1],col+2)):
                if self.regions[i,j]!=color:
                    border=True
        if border:
            self.grid[(i,j)]='water'

    def fill(self,location):

        (row,col)=location

        color=self.regions[row,col]

        filled=False
        for i in range(max(0,row-1),min(self.regions.shape[0],row+2)):
            for j in range(max(0,col-1),min(self.regions.shape[1],col+2)):
                if self.regions[i,j]==0:
                    self.regions[i,j]=color
                    #self.grid[i,j]=color
                    filled=True
        
        return filled

    '''
        
                    
    def step(self,action):
        terminated=False

        rows=GRID_DIM[0]
        cols=GRID_DIM[1]

        if action=='up':
            if self.y+1<cols and self.grid[(self.x,self.y+1)]!='water':
                self.y+=1

        if action=='down':
            if self.y-1>=0 and self.grid[(self.x,self.y-1)]!='water':
                self.y-=1

        if action=='left':
            if self.x-1>=0 and self.grid[(self.x-1,self.y)]!='water':
                self.x-=1

        if action=='right':
            if self.x+1<rows and self.grid[(self.x+1,self.y)]!='water':
                self.x+=1

        if action=='1' or action=='brick':
            if self.y-1>0 and self.grid[(self.x,self.y-1)]!='water':
                self.grid[(self.x,self.y-1)]='brick'

        if action=='2' or action=='door':
            if self.y-1>0 and self.grid[(self.x,self.y-1)]!='water':
                self.grid[(self.x,self.y-1)]='door'

        if action=='3' or action=='red_tile':
            if self.y-1>0 and self.grid[(self.x,self.y-1)]!='water':
                self.grid[(self.x,self.y-1)]='red_tile'

        if action=='4' or action=='land':
            if self.y-1>0 and self.grid[(self.x,self.y-1)]!='water':
                self.grid[(self.x,self.y-1)]='land'


        if action=='bridge':
            self.build_bridge()

        if action=='reset':
            self.__init__()
            terminated=True

        self.x=max(0,self.x)
        self.y=max(0,self.y)
        self.t+=1

        return terminated

    def build_bridge(self):

        row=self.x
        col=self.y

        for i in range(max(0,row-1),min(self.regions.shape[0],row+2)):
            for j in range(max(0,col-1),min(self.regions.shape[1],col+2)):
                if self.grid[(i,j)]=='water':
                    self.grid[(i,j)]='bridge'

    def match(self,x,y,c):
        if self.grid[x,y]==c or self.grid[x,y]-4==c or self.grid[x,y]==c-4:
            return True
        else:
            return False
    
    def get_rgb_matrix(self,scale=1):
        rows=GRID_DIM[0]
        cols=GRID_DIM[1]

        scale=int(scale)

        rgb_matrix=np.zeros((rows*scale,cols*scale,3))

        for x_g in range(0, rows):
            for y_g in range(0, cols):
                val=self.grid[(x_g,y_g)]
                x=scale*x_g
                y=scale*y_g
                if val=='empty':
                    rgb_matrix[x:x+scale,y+scale,:]=[0,0,0]

                elif val=='door':
                    rgb_matrix[x:x+scale,y:y+scale,:]=[50,50,50]

                elif val=='brick':
                    rgb_matrix[x:x+scale,y:y+scale,:]=[150,100,50]

                elif val=='red_tile':
                    rgb_matrix[x:x+scale,y:y+scale,:]=[150,50,30]

                elif val=='land':
                    rgb_matrix[x:x+scale,y:y+scale,:]=[0,50,0]

                elif val=='water':
                    rgb_matrix[x:x+scale,y:y+scale,:]=[50,50,100]

                elif val=='bridge':
                    rgb_matrix[x:x+scale,y:y+scale,:]=[100,100,100]

                elif val=='target':
                    rgb_matrix[x:x+scale,y:y+scale,:]=[50,90,25]

                if (x_g,y_g)==(self.x,self.y):
                    if self.is_subgoal_state_indicator and self.view_subgoals:
                        rgb_matrix[x:x+scale,y:y+scale,:]=[100,100,250]
                    else:
                        rgb_matrix[x:x+scale,y:y+scale,:]=[200,200,200]

        return rgb_matrix

    def is_goal_state(self):
        rows=GRID_DIM[0]
        cols=GRID_DIM[1]

        equal=True
        for i in range(0, rows):
            for j in range(0, cols):
                if self.goal_mask[(i,j)]!='water' and self.grid[(i,j)]!=self.goal_mask[(i,j)]:
                    equal=False

        return equal

class AutoConstruct:
    def __init__(self,world,padding=None):
        self.world=world
        self.padding=padding
        

        self.mask=world.grid.copy()

        #self.subtasks=['construct_house','construct_house','construct_house','move_to_site_centroid']
        self.subtasks=['construct_house']

        if False and padding=='random_moves':

            for i in range(4):
                self.subtasks=['random_move']+self.subtasks+['random_move']

        self.complete=True
        self.generate_masks()

        self.init_world=deepcopy(world)
        #self.init_steps=self.get_steps_to_goal()
        self.init_steps=None

    def get_steps_from_init(self):
        if self.init_steps==None:
            trials=[]
            for i in range(30):
                trials.append(self.get_steps_to_goal())
            self.init_steps=sum(trials)/(1.0*len(trials))

        return self.init_steps

    def is_subgoal_state(self):
        init_steps=self.get_steps_from_init()
        trials=[]
        for i in range(2):
            trials.append(self.get_steps_to_goal())
        cur_steps=sum(trials)/(1.0*len(trials))

        #print(cur_steps,init_steps)
        return cur_steps<init_steps

    def get_steps_to_goal(self):

        temp_world=deepcopy(self.world)
        temp_auto_construct=AutoConstruct(temp_world,'random_moves')

        count=0
        terminated=False
        reward=0.0

        while reward==0.0:
            action,annotation,reward,terminated=temp_auto_construct.auto_step()
            count+=1

        return count

    def generate_masks(self):
        image=HOUSE_IMAGE

        h=len(image[0])
        w=len(image)

        self.masks=[]
        current_mask=self.world.grid.copy()
        self.masks=list(self.world.checkpoints)
    
    def auto_step(self):

        annotation='none'
        action='none'

        if self.complete:
            if len(self.subtasks)>0:
                self.subtask=self.subtasks.pop()
                self.complete=False
                self.pstep=0
            else:
                action='reset'
                self.complete=False

        else:

            if self.subtask=='move_to_site_centroid':
                action=self.subtask_move_to_site_centroid()

            if self.subtask=='construct_house':
                action,annotation=self.subtask_construct_house()

            if self.subtask=='random_move':
                action=self.subtask_random_move()

            if self.complete:
                annotation="completed: "+str(self.subtask)
                action='none'
                terminal=True

                if self.world.is_goal_state():
                    reward=1.0
                else:
                    reward=0.0

                return action, annotation, reward, terminal

        terminal=self.world.step(action)

        if self.world.is_goal_state():
            reward=1.0
        else:
            reward=0.0

        if action=='reset':
            self.__init__(self.world,padding=self.padding)

        return action, annotation, reward, terminal

    def subtask_random_move(self):

        if self.pstep==0:
            self.ptarget=random.choice(list(self.world.centroids.items()))[1]
            self.pstep+=1

        if self.pstep>=1:

            (x,y)=self.ptarget
            action=self.calc_action(x,y,'move')
            self.pstep+=1

        if action==None:
            self.complete=True

        return action

    def subtask_move_to_site_centroid(self):

        action=self.calc_action(self.world.target_x,self.world.target_y,'move')

        if action==None:
            self.complete=True

        return action

    def subtask_construct_house(self):

        action='none'
        annotation='none'

        if self.pstep==0:

            self.dests=[]
            self.first_construct=False
            self.cur_mask=self.masks.pop()
            self.dests=self.get_dests(self.cur_mask)
            self.cur_mask=self.masks.pop()
            self.dests=self.get_dests(self.cur_mask)+self.dests

            self.pstep+=1

        if self.pstep==1:

            if len(self.dests)>0:
                self.dest=self.dests.pop()
                self.pstep+=1
            else:
                self.dests=self.get_dests(self.cur_mask)+self.dests
                if len(self.dests)>0:
                    self.dest=self.dests.pop()
                    self.pstep+=1
                else:
                    self.complete=True
                    return action, annotation

        if self.pstep>=2:
            action=self.calc_action(self.dest[0],self.dest[1],self.dest[2])
            if action==None:
                self.pstep=1
                self.subtask_construct_house()

        if action=='door' and not self.first_construct:
            annotation='begin: construct_house'
            self.first_construct=True

        return action, annotation

    def get_dests(self,mask):
        changes=[]
        for i in range(GRID_DIM[0]):
            for j in range(GRID_DIM[1]):
                if mask[(i,j)]!=self.world.grid[(i,j)] and mask[(i,j)]!='water':
                    changes.append((i,j,mask[(i,j)]))
        random.shuffle(changes)
        return changes

    def mask_image(self,x,y,image,mask):

        h=len(image[0])
        w=len(image)

        changes=[]

        if x+w<GRID_DIM[0] and y+h<GRID_DIM[1]:
            for i in range(0,w):
                for j in range(0,h):
                    if image[i][j]!=None:
                        self.mask[(x+i,y+j)]=image[i][j]
                        changes.append((x+i,y+j,image[i][j]))
        return changes

    def calc_action(self,dx,dy,color):
        action='none'
        dy=dy+1
        if abs(dx-self.world.x)>abs(dy-self.world.y):
            if dx<self.world.x:
                if self.world.x-1>=0 and self.world.grid[(self.world.x-1,self.world.y)]!='water':
                    return 'left'
                else:
                    return 'bridge'
            elif dx>self.world.x:
                if self.world.x+1<GRID_DIM[0] and self.world.grid[(self.world.x+1,self.world.y)]!='water':
                    return 'right'
                else:
                    return 'bridge'
        else:
            if dy<self.world.y:
                if self.world.y-1>=0 and self.world.grid[(self.world.x,self.world.y-1)]!='water':
                    return 'down'
                else:
                    return 'bridge'

            elif dy>self.world.y:
                if self.world.y+1<GRID_DIM[1] and self.world.grid[(self.world.x,self.world.y+1)]!='water':
                    return 'up'
                else:
                    return 'bridge'

        if color!='move' and self.world.grid[(self.world.x,self.world.y-1)]!=color:
            return color
        else:
            return None

    def house_count(self):

        h=len(HOUSE_IMAGE[0])
        w=len(HOUSE_IMAGE)

        count=0 

        for x in range(0,GRID_DIM[0]):
            for y in range(0,GRID_DIM[1]):
                if x+w<GRID_DIM[0] and y+h<GRID_DIM[1]:
                    found=True
                    for i in range(0,w):
                        for j in range(0,h):
                            if HOUSE_IMAGE[i][j]!=None and HOUSE_IMAGE[i][j]!=self.world.grid[(x+i,y+j)]:
                                found=False
                    if found:
                        count+=1
        return count

class Construct:
    def __init__(self,auto_padding=None):
        self.name='construct'
        self.world=World()
        self.auto_construct=AutoConstruct(self.world,padding=auto_padding)

    def export_state(self):
        return { \
            'grid':self.world.grid.copy(),\
            'x':self.world.x,\
            'y':self.world.y,\
            'checkpoints':self.world.checkpoints.copy(),\
            'goal_mask':self.world.goal_mask,
            'subtasks':self.auto_construct.subtasks.copy(),\
            'subtask' :self.auto_construct.subtask
        }

    def load_state(self,state_dict):
        self.world.grid=state_dict['grid'].copy()
        self.world.x=state_dict['x']
        self.world.y=state_dict['y']
        self.world.checkpoints=state_dict['checkpoints'].copy()
        self.world.goal_mask=state_dict['goal_mask'].copy()
        self.auto_construct.subtasks=state_dict['subtasks']
        self.auto_construct.subtask=state_dict['subtask']

    def reset(self):
        self.__init__()

    def step(self,action):
        self.world.step(action)

    def auto_step(self):
        state0=self.world.get_rgb_matrix()
        action,annotation,reward,terminated=self.auto_construct.auto_step()
        state1=self.world.get_rgb_matrix()
        return state0,state1,action,annotation,reward,terminated,self.world.t

    def random_step(self):
        actions=['up','down','left','right','1','2','3','4']
        action=random.choice(actions)
        state0=self.world.get_rgb_matrix()
        reward,terminated=self.world.step(action)
        annotation=None
        state1=self.world.get_rgb_matrix()
        return state0,state1,action,annotation,reward,terminated,self.world.t

    def set_auto_padding(self,padding):
        self.auto_construct=AutoConstruct(self.world,padding=padding)

    def draw(self,screen):
        rgb_matrix=self.world.get_rgb_matrix()
        draw_rgb_matrix(screen,rgb_matrix) 

class ConstructEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(8)
        self.observation_space = \
            spaces.Box(low=0, high=255, shape=(GRID_DIM[0], GRID_DIM[1], 3))
        self.house_image = \
            [[None, 'red_tile', 'red_tile', 'red_tile', None],
             ['red_tile', 'red_tile', 'red_tile', 'red_tile', 'red_tile'],
             [None, 'brick', 'brick', 'brick', None],
             [None, 'brick', 'door', 'brick', None]]
        self.num_envs = 1
        self.reset()

def main():

    ''' 
    world=World()
    auto_construct=AutoConstruct(world,'random_moves')
    '''
    simulation=Construct('random_moves')

    world=simulation.world
    auto_construct=simulation.auto_construct
    
    pygame.init()

    window_size=(800, 600)

    window=Window(window_size)
    screen=window.surface
    window.update_title(__file__)

    myclock=pygame.time.Clock()
        
    framerate_limit=4000

    t = 0.0
    finished=False
    mode='auto'
    
    while not finished:
        
        window.surface.fill(THECOLORS["black"])
        dt=float(myclock.tick(framerate_limit) * 1e-3)
        
        command=window.get_user_input()

        if command:
            pass

        if command=='reset':
            world=World()
            auto_construct=AutoConstruct(world,padding='random_moves')
        
        elif (command=='quit'):
            finished=True
            continue
            
        elif (command!=None):
            pass

        if command=='m':
            if mode=='user':
                mode='auto'
            elif mode=='auto':
                mode='user'
        
        if command=='count':
            trials=[]
            for i in range(5):
                trials.append(auto_construct.get_steps_to_goal())
            avg=sum(trials)/(1.0*len(trials))
            print("Steps to goal: ",avg)

        if command=='subgoal':
            is_subgoal=auto_construct.is_subgoal_state()
            world.is_subgoal_state_indicator=is_subgoal
            print("SG: ",is_subgoal)

        if command=='save':
            save_state=simulation.export_state()
            print("State saved.")

        if command=='load':
            simulation.load_state(save_state)

        if command=='view_subgoals':
            world.view_subgoals=not world.view_subgoals

        if mode=='auto':
            action,annotation,reward,terminated=auto_construct.auto_step()
            '''
            if annotation!='none':
                print(annotation)
            '''

        if     command=='up'\
            or command=='down'\
            or command=='left'\
            or command=='right'\
            or command=='1'\
            or command=='2'\
            or command=='3'\
            or command=='bridge':
                world.step(command)

        if world.view_subgoals:
            is_subgoal=auto_construct.is_subgoal_state()
            world.is_subgoal_state_indicator=is_subgoal

        s0=world.get_rgb_matrix()
        draw_rgb_matrix(screen,s0) 
        
        t+=dt
        pygame.display.flip()


if __name__=="__main__":
    main()
