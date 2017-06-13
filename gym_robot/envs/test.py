import unittest2 as unittest
from obstacles import Robot, Obstacle


class TestRobotMethods(unittest.TestCase):

    def setUp(self):
        self.robot = Robot([300, 300], 50, 30)

    def test_turn_left(self):
        currentAngle = self.robot.get_rotation()
        self.robot.turn_left()
        self.assertEquals(self.robot.get_rotation(), currentAngle + 1)

    def test_turn_right(self):
        currentAngle = self.robot.get_rotation()
        self.robot.turn_right()
        self.assertEquals(self.robot.get_rotation(), currentAngle - 1)

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

    def test_line_intersect(self):
        p1 = [0, 0]
        p2 = [4, 4]
        p3 = [4, 0]
        p4 = [0, 4]
        self.assertTrue(self.robot.line_intersect([p1, p2], [p3, p4]))

    def test_line_intersect2(self):
        p1 = [0, 0]
        p2 = [4, 4]
        p3 = [4, 0]
        p4 = [0, 4]
        self.assertTrue(self.robot.line_intersect([p2, p1], [p4, p3]))

    def test_line_intersect_no_intersection(self):
        p1 = [6, -2]
        p2 = [2, 2]
        p3 = [-2, 0]
        p4 = [2, 4]
        self.assertFalse(self.robot.line_intersect([p1, p2], [p3, p4]))

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


if __name__ == '__main__':
    unittest.main()
