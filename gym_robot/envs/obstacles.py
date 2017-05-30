import numpy as np
class Obstacle(object):
    def __init__(self,position, width, height):
        self.position = position
        self.width = width
        self.height = height
    def move(self,x,y):
        self.position[0] = x
        self.position[1] = y
    def collision(self,obj):
        pass


class Robot(Obstacle):
    def __init__(self,position, width, height):
        Obstacle.__init__(self,position,width,height)
        self.angle = 0
        self.speed=0.5
        self.turn_speed = 1
    def get_postion(self):
        return self.position
    def get_rotation(self):
        return self.angle
    def move_forward(self):
        self.position[0] += np.sin(np.pi/2 - self.angle* np.pi/180)*self.speed
        self.position[1] += np.sin(self.angle * np.pi/180)*self.speed
    def turn_left(self):
        self.angle += self.turn_speed
    def turn_right(self):
        self.angle -= self.turn_speed