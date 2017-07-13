import numpy as np
import linalg_helper
from shapely.geometry import Polygon, LineString
from shapely import speedups

class Obstacle(object):
    def __init__(self, position, width, height):
        self.position = position
        self.width = width
        self.angle = 0
        self.height = height
        if speedups.available:
            speedups.enable()

    def move(self, x, y):
        self.position[0] = x
        self.position[1] = y

    def collision(self, obj):
        pass

    def get_drawing(self):
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        return [(l, b), (l, t), (r, t), (r, b)]
    
    def get_drawing_static_position(self):
        x = self.get_postion()[0]
        y = self.get_postion()[1]
        l, r, t, b = -self.width / 2 + x, self.width / 2 + x, self.height / 2 + y, -self.height / 2 + y
        return [(l, b), (l, t), (r, t), (r, b)]

    def get_postion(self):
        return self.position

    def get_corners(self):
        c1, c2, c3, c4 = [0, 0], [0, 0], [0, 0], [0, 0]
        centerPoint = self.get_postion()
        c1[0] = centerPoint[0] - self.width / 2
        c1[1] = centerPoint[1] - self.height / 2

        c2[0] = centerPoint[0] - self.width / 2
        c2[1] = centerPoint[1] + self.height / 2

        c3[0] = centerPoint[0] + self.width / 2
        c3[1] = centerPoint[1] + self.height / 2

        c4[0] = centerPoint[0] + self.width / 2
        c4[1] = centerPoint[1] - self.height / 2

        return[c1, c2, c3, c4]


class Robot(Obstacle):
    def __init__(self, position, width, height):
        Obstacle.__init__(self, position, width, height)
        self.angle = 0
        self.speed = 2.5
        self.turn_speed = 1

    def get_position(self):
        return self.position

    def get_rotation(self):
        return self.angle

    def move_forward(self):
        self.position[0] += np.sin(np.pi / 2 -
                                   self.angle * np.pi / 180) * self.speed
        self.position[1] += np.sin(self.angle * np.pi / 180) * self.speed

    def turn_left(self):
        self.angle += self.turn_speed

    def turn_right(self):
        self.angle -= self.turn_speed

    def collision(self, obj):
        if type(obj) is Obstacle:
            #xmin, ymin, xmax, ymax = self.get_rect_min_max(self.get_rotated_corners())
            #x2min, y2min, x2max, y2max = self.get_rect_min_max(  obj.get_corners())
            
            robot_corners = self.get_rotated_corners()
            obj_corners = obj.get_corners()
            p1 = Polygon(robot_corners )
            p2 = Polygon(obj_corners)
            ret = p1.intersects(p2)
            
            return ret


    def get_rotated_corners(self):
        corners = self.get_corners()
        for i in xrange(len(corners)):
            angleInRad = self.angle * np.pi / 180
            translatedCorner = np.subtract(self.position, corners[i])
            rotatedX = translatedCorner[0] * np.cos(
                angleInRad) - translatedCorner[1] * np.sin(angleInRad)
            rotatedY = translatedCorner[0] * np.sin(
                angleInRad) + translatedCorner[1] * np.cos(angleInRad)
            corners[i] = [rotatedX + self.position[0],
                          rotatedY + self.position[1]]
        return corners

    def get_rotated_corner(self, corner, angle, mid):
        angleInRad = angle * np.pi / 180
        translatedCorner = np.subtract(mid, corner)
        rotatedX = translatedCorner[0] * \
            np.cos(angleInRad) - translatedCorner[1] * np.sin(angleInRad)
        rotatedY = translatedCorner[0] * \
            np.sin(angleInRad) + translatedCorner[1] * np.cos(angleInRad)
        return [rotatedX + mid[0], rotatedY + mid[1]]

    def usSensors(self,objectList):
        us_angles = [-20,0,20]
        mins = [0,0,0]
        interections = [0,0,0]
        start = [0,0,0]
        for i in range(len(us_angles)):
            mins[i],interections[i],start[i] = self.rayCast(objectList,us_angles[i])
        return mins,interections,start[1]

    # returns distance to nearest object
    def rayCast(self,objectList,dirAngle=0,vec_range=255):
        #relative position of ultrasonic sensor
        angleInRad = (self.angle + dirAngle) * np.pi / 180
        dirVec = np.array([np.cos(angleInRad),np.sin(angleInRad)])
        posRot = np.add(dirVec * self.width/2, self.get_position())
        direction = dirVec*vec_range
        p2 = posRot + direction
        line = LineString([posRot,p2])
        minimum = vec_range
        min_intersection = [0,0]
        for obj in objectList:
            obj_pol = Polygon(obj.get_corners())
            intersection = obj_pol.intersection(line)
            if type(intersection) is LineString:
                intersection = list(obj_pol.intersection(line).coords)      
                distances = np.linalg.norm(intersection-posRot,axis = 1 )
                index = np.argmin(distances)
                length = distances[index]
                if length < minimum:
                    minimum = length
                    min_intersection = np.array(intersection[index])
        return minimum,min_intersection,posRot             
        # send Ray
        # minimum = vec_range
        # min_intersection = [[0,0]]
        # for obj in objectList:
        #     corners = obj.get_corners()
        #     segments = self.get_segments(corners)
        #     for s in segments:
        #         intersection = self.intersects(posRot,direction,s[0],s[1])
        #         length = np.linalg.norm(intersection-posRot)
        #         if length < minimum:
        #             minimum = length
        #             min_intersection = intersection
        # return minimum,min_intersection[0],posRot

    def get_segments(self,corners):
        corners = np.array(corners)
        v1 = np.array([corners[0],corners[1]])
        v2 = np.array([corners[1],corners[2]])
        v3 = np.array([corners[2],corners[3]])
        v4 = np.array([corners[3],corners[0]])  
        return np.array([v1,v2,v3,v4])      

    def get_rect_min_max(self, corners):
        xmax, ymax = np.max(corners, axis=0)
        xmin, ymin = np.min(corners, axis=0)
        return xmin, ymin, xmax, ymax

    def intersects(self, rayOrigin, rayDirection, p1 ,p2):
        rayOrigin = np.array(rayOrigin, dtype=np.float)
        rayDirection = np.array(self.norm(rayDirection), dtype=np.float)
        point1 = np.array(p1, dtype=np.float)
        point2 = np.array(p2, dtype=np.float)
        v1 = rayOrigin - point1
        v2 = point2 - point1
        v3 = np.array([-rayDirection[1], rayDirection[0]])
        t1 = np.cross(v2, v1) / np.dot(v2, v3)
        t2 = np.dot(v1, v3) / np.dot(v2, v3)
        if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
            return np.array([rayOrigin + t1 * rayDirection])
        return np.array([np.inf,np.inf])

    def magnitude(self, vector):
        return np.sqrt(np.dot(np.array(vector),np.array(vector)))

    def norm(self, vector):
        return np.array(vector)/self.magnitude(np.array(vector))

    def rotate_point(self,point,angle):
        angleInRad = angle * np.pi / 180
        point[0] = point[0] * np.cos(angleInRad) - point[1] * np.sin(angleInRad)
        point[1] = point[0] * np.sin(angleInRad) + point[1] * np.cos(angleInRad)
        return point