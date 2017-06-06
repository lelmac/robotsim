import numpy as np
class Obstacle(object):
    def __init__(self,position, width, height):
        self.position = position
        self.width = width
        self.angle = 0
        self.height = height
    def move(self,x,y):
        self.position[0] = x
        self.position[1] = y
    def collision(self,obj):
        pass
    def get_drawing(self):
        l,r,t,b = -self.width/2,self.width/2,self.height/2,-self.height/2
        return [(l,b), (l,t), (r,t), (r,b)]
    def get_postion(self):
        return self.position
        
class Robot(Obstacle):
    def __init__(self,position, width, height):
        Obstacle.__init__(self,position,width,height)
        self.angle = 0
        self.speed=0.5
        self.turn_speed = 1
    def get_position(self):
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

        if type(obj) is Obstacle:
            lines = self.get_collision_lines(self)
            lines2 = self.get_collision_lines(obj)    
            for i in xrange(len(lines)):
                for j in xrange(len(lines2)):
                    if self.line_intersect(lines[i],lines2[j]):
                        print("collision")

        
    def get_rotated_corner(self,corner,angle,mid):
        angleInRad = angle * np.pi/180
        translatedCorner = np.subtract(mid,corner)
        rotatedX = translatedCorner[0]*np.cos(angleInRad) - translatedCorner[1]*np.sin(angleInRad)
        rotatedY = translatedCorner[0]*np.sin(angleInRad) + translatedCorner[1]*np.cos(angleInRad)
        return [rotatedX + mid[0],rotatedY + mid[1]]

    def get_corners(self):
        c1,c2,c3,c4 = [0,0],[0,0],[0,0],[0,0]
        centerPoint = self.get_postion()
        c1[0] = centerPoint[0] - self.width/2
        c1[1] = centerPoint[1] - self.height/2
            
        c2[0] = centerPoint[0] - self.width/2
        c2[1] = centerPoint[1] + self.height/2

        c3[0] = centerPoint[0] + self.width/2
        c3[1] = centerPoint[1] + self.height/2

        c4[0] = centerPoint[0] + self.width/2
        c4[1] = centerPoint[1] - self.height/2

        return[c1,c2,c3,c4]

    def get_collision_lines(self,obj):
        corners = self.get_corners()

        left_down = self.get_rotated_corner(corners[0],obj.angle,obj.position)
        right_down = self.get_rotated_corner(corners[1],obj.angle,obj.position)
        right_up = self.get_rotated_corner(corners[2],obj.angle,obj.position)
        left_up = self.get_rotated_corner(corners[3],obj.angle,obj.position)

        return [[left_down,left_up],[left_down,right_down],[left_up,right_up],[right_up,left_down]]

    def perp(self,a) :
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    def line_intersect(self,a,b) :
        a = np.array(a)
        b = np.array(b)
        da = a[1]-a[0]
        db = b[1]-b[0]
        dp = a[0]-b[0]
        dap = self.perp(da)
        denom = np.dot( dap, db)
        num = np.dot( dap, dp )
        intersection = (num / denom.astype(float))*db + b[0]
        if not np.isnan(intersection).any():
            return np.linalg.norm(intersection- a[0]) >= np.linalg.norm(da)
        return False