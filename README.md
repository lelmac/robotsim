# Simulator for EV3 as Environment for OpenAI Gym

## 4 Environments:
* AutonomousRobot-v0: Discrete Action Space, Infraredsensor and Ultrasonic Sensor, Random Spawns
* AutonomousRobot-v1: Continuous Action Space, Infraredsensor and Ultrasonic Sensor, Random Spawns
* AutonomousRobot-v2: Discrete Observation Space, Discrete Action Space (for Q-Learning), Infrared & Ultrasonic Sensor
* AutonomousRobot-v3: Discrete Action Space, Target Position, Random Spawns, 3 Distance Sensors (Basically Scenario 3 from the Thesis)

Each Environment can of course be customized indivually.

## Setup
1. Clone the Repo
2. Install OpenAI Gym: https://github.com/openai/gym
3. Run:
```
  sudo pip install -r requirements
  sudo python setup.py install
```
 
OR: Use the Dockerfile to create a Container

Then you should be ready to use the Environments by import gym_robot:
```
  import gym_robot
```

## Misc
The Repository comes with a few extas:
* Few Agents: DDPG, DQN, DQN (With Keras-RL), Q-Learning, A3C(not really threaded), Keyboard Agent (can be handy when designing environments)
* A Dockerfile
* 2 Scripts for visualizing logfiles:

  * for keras-rl logs:
  ```
  visualize.py filename.ending
  ```
  * for csv files:
  ```
  cs.py filename.ending 
  ```
* Some pretrained Models

## Simulation:
* The simulation is located in gym_robot/envs/obstacles.py
* Got 2 classes: one for obstacles and one for the robot itself
* The Methods for the sensors are in robot class
* Rendering is done by OpenAI Gym Renderer in the specific environment
* test.py contains unit tests for the classes
