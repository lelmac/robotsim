import unittest2 as unittest
from obstacles import Robot,Obstacle
class TestRobotMethods(unittest.TestCase):

    def setUp(self):
        self.robot = Robot([300,300],50,30)


    def test_turn_left(self):
        currentAngle = self.robot.get_rotation()
        self.robot.turn_left()
        self.assertEquals(self.robot.get_rotation(),currentAngle +1)
    def test_turn_right(self):
        currentAngle = self.robot.get_rotation()
        self.robot.turn_right()
        self.assertEquals(self.robot.get_rotation(),currentAngle -1)

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

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()