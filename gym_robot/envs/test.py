import unittest2 as unittest
from shapely.geometry import Polygon
import linalg_helper
from obstacles import Robot, Obstacle
import numpy as np


class TestRobotMethods(unittest.TestCase):

    def setUp(self):
        self.robot = Robot([300, 300], 50, 30)

    def test_turn_left(self):
        currentAngle = self.robot.get_rotation()
        self.robot.turn_left()
        self.assertEquals(self.robot.get_rotation(), currentAngle + self.robot.turn_speed)

    def test_turn_right(self):
        currentAngle = self.robot.get_rotation()
        self.robot.turn_right()
        self.assertEquals(self.robot.get_rotation(), currentAngle - self.robot.turn_speed)

    def test_move_forward_straight(self):
        self.robot.angle = 0
        self.robot.speed = 1
        pos = self.robot.get_position()
        self.robot.move_forward()
        pos[0] = pos[0] + 1
        self.assertEquals(self.robot.get_postion(), pos)

    def test_move_forward_angle(self):
        self.robot.angle = 45
        self.robot.speed = 1
        pos = self.robot.get_position()
        self.robot.move_forward()
        increment = 0.5**0.5
        pos[0] = pos[0] + increment
        pos[1] = pos[1] + increment
        self.assertEquals(self.robot.get_postion(), pos)

    def test_collision(self):
        obstacle = Obstacle([350, 350], 150, 150)
        self.assertTrue(self.robot.collision(obstacle))

    def test_collision_on_line(self):
        self.robot = Robot([300, 300], 100, 100)
        obstacle = Obstacle([400, 300], 100, 100)
        self.assertTrue(self.robot.collision(obstacle))

    def test_collision_on_line_slim(self):
        self.robot = Robot([300, 300], 100, 50)
        obstacle = Obstacle([400, 300], 100, 100)
        self.assertTrue(self.robot.collision(obstacle))

    def test_collision_rotation(self):
        self.robot = Robot([300, 300], 100, 100)
        self.robot.angle = 46
        obstacle = Obstacle([350, 350], 150, 150)
        self.assertTrue(self.robot.collision(obstacle))

    def test__edge_collision_rotation(self):
        self.robot = Robot([300, 300], 50, 50)
        self.robot.angle = 45
        obstacle = Obstacle([350, 350], 100, 50)
        self.assertTrue(self.robot.collision(obstacle))

    def test_no_collision(self):
        self.robot = Robot([300, 300], 50, 30)
        obstacle = Obstacle([500, 500], 30, 30)
        self.assertFalse(self.robot.collision(obstacle))
    
    def test_line_intersect3(self):
        p1 = [0, 0]
        p2 = [4, 4]
        p3 = [4, 0]
        p4 = [0, 4]
        intersection = np.array([2,2])
        
        self.assertEqual(self.robot.intersects(p1, p2, p3, p4).all(),intersection.all())

    def test_line_intersect4(self):
        p1 = [1, 1]
        p2 = [4, 4]
        p3 = [4, 0]
        p4 = [0, 4]
        intersection = np.array([2,2])
        self.assertEqual(self.robot.intersects(p2, p1, p4, p3).all(),intersection.all())
    
    def test_ray_casting(self):
        self.robot = Robot([300, 300], 50, 50)
        self.robot.angle = 0
        obstacle = Obstacle([400, 300], 50, 50)
        r,p,p2 = self.robot.rayCast([obstacle])
        self.assertEqual(r,50.0)
    
    def test_ray_casting_nohit(self):
        self.robot = Robot([300, 300], 50, 50)
        self.robot.angle = 0
        obstacle = Obstacle([300, 400], 50, 50)
        r,p,p2 = self.robot.rayCast([obstacle])
        self.assertEqual(r,255)

    def test_coll_rotated(self):
        v1 = [[425.643330675622, 247.83978898773091], [425.643330675622, 277.83978898773091], [475.643330675622, 277.83978898773091], [475.643330675622, 247.83978898773091]]
        v2 = [[475, 275], [475, 325], [525, 325], [525, 275]]
        p1 = Polygon(v1)
        p2 = Polygon(v2)
        self.assertTrue(p1.intersects(p2))
        #self.assertFalse(linalg_helper.separating_axis_theorem(v1,v2))


if __name__ == '__main__':
    unittest.main()
