class Obstacle(object):
    def __init__(self,position, width, height):
        self.position = position
        self.width = width
        self.height = height
    def move(self,x,y):
        pass
    def collision(self,obj):
        pass


class Robot(Obstacle):
    def __init__(self,position, width, height):
        Obstacle.__init__(self)
    def move_forward(self):
        pass
    def turn_left(self):
        pass
    def turn_right(self):
        pass