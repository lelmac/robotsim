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
    def collision(self,obj):
        left_down=[]
        left_up = []
        right_up = []
        right_down=[]
        if type(obj) is Obstacle:
            left_down[0] = self.position[0] - self.width/2
            left_down[1] = self.position[1] - self.height/2

            left_up[0] = self.position[0] - self.width/2
            left_up[1] = self.position[1] + self.height/2

            right_up[0] = self.position[0] + self.width/2
            right_up[1] = self.position[1] + self.height/2

            right_down[0] = self.position[0] + self.width/2
            right_down[1] = self.position[1] - self.height/2

            left_down = get_rotated_corner(left_down,self.angle,self.position)
            right_down = get_rotated_corner(right_down,self.angle,self.position)
            right_up = get_rotated_corner(right_up,self.angle,self.position)
            left_up = get_rotated_corner(left_up,self.angle,self.position)


            v1 = np.subtract(left_up,left_down)
            v2 = np.subtract(right_up, right_down)
            v3 = np.subtract(left_up,right_up)
            v4 = np.subtract(left_down, right_down)

        def get_rotated_corner(corner,angle,mid):
            angleInRad = angle * np.pi/180
            translatedCorner = np.subtract(mid,corner)
            rotatedX = translatedCorner[0]*np.cos(angleInRad) - translatedCorner[1]*np.sin(angleInRad)
            rotatedY = translatedCorner[0]*np.sin(angleInRad) + translatedCorner[1]*np.cos(angleInRad)
            return [rotatedX + mid[0],rotatedY + mid[1]]