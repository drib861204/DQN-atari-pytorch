"""
Title: pendulum_pygame
Author: [jadenhensley](https://github.com/jadenhensley)
Last modified: 2021/10/18
Description: Pendulum project, built using pygame and math modules.

Title: wheelPole
Author: [aimetz](https://github.com/aimetz)
Last modified: 2021/04/20

Title: gym/gym/envs/classic_control/pendulum.py
Author: [openai](https://github.com/openai)
Last modified: 2021/10/31
"""
import pygame
from math import pi, sin, cos
import numpy as np

import gym
from gym import spaces, logger
#from gym.utils import seeding
#from gym.utils.seeding import RandomNumberGenerator


class Pendulum(gym.Env):
    def __init__(self, rend, seed):
        #self.np_random = np.random.seed(seed)
        #for i in range(10):
        #    print(np.random.uniform(low=-3.5, high=3.5)*pi/180)

        self.theta_rod = 0
        self.theta_wheel = 0
        self.theta_rod_dot = 0
        self.theta_wheel_dot = 0
        self.len_rod = 0.35
        self.len_wheel = 0.5
        self.mass_rod = 20
        self.rad_out = 0.05
        self.rad_in = 0.045
        self.t = 0.01
        self.rho = 7870
        self.mass_wheel = (self.rad_out**2-self.rad_in**2)*pi*self.t*self.rho
        #self.momentum_rod = self.mass_rod*(2*self.len_rod)**2/33
        self.momentum_rod=(2*self.len_rod)*self.mass_rod*(2*self.len_rod)**2/12+(1-(2*self.len_rod))*self.mass_rod*(1-(2*self.len_rod))**2/12
        self.momentum_wheel = self.mass_wheel*(self.rad_out**2+self.rad_in**2)/2
        self.dt = 0.001
        self.gravity = 9.81
        self.max_q1 = 15*pi/180
        self.max_q1dot = 0.3 #initial q1_dot default 0.3? to be verified
        self.wheel_max_speed = 20
        self.max_torque = 15
        self.torque = 0

        l1 = self.len_rod
        l2 = self.len_wheel
        m1 = self.mass_rod
        m2 = self.mass_wheel
        I1 = self.momentum_rod
        I2 = self.momentum_wheel

        self.Ip = m2*l2**2+I1+I2 + 2*l1*m1*l1**2+(1-2*l1)*m1*(0.5*(1-2*l1)+2*l1)**2
        self.mbarg = (m1*l1+m2*l2)*self.gravity

        high = np.array([2*self.max_q1, self.max_q1dot, self.wheel_max_speed], dtype=np.float32)
        #self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        #print(self.mass_wheel)
        #print(self.momentum_rod)
        #print(self.momentum_wheel)

        width = 800
        height = 600
        self.origin_x = width//2
        self.origin_y = height//2
        self.POS = np.array([self.origin_x, self.origin_y])

        if rend:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Pendulum Simulation")
            pygame.font.init()
            self.debug_font = pygame.font.SysFont('Bauhuas 93', 30)
            self.hint_font = pygame.font.SysFont('Bauhaus 93', 26)
            #print("font")


    def reset(self, saved, seed, return_info=False):

        #self.np_random = np.random.seed(seed)
        #super().reset(seed=seed)
        #self._np_random, seed = seeding.np_random(seed)

        reset_angle = 10*pi/180

        if saved == None:
            reset_high = np.array([reset_angle, self.max_q1dot, self.wheel_max_speed])
            #self.state = self.np_random.uniform(low=-reset_high, high=reset_high)
            self.state = np.random.uniform(low=-reset_high, high=reset_high)
        else:
            self.state = np.array([reset_angle, 0, 0], dtype=np.float32)
            # self.state = np.array([0, self.max_q1dot, 0],dtype=np.float32)

        self.last_u = None

        if not return_info:
            return self.state
        else:
            return self.state, {}


    def render(self, eval_run):
        #torque = action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


        SCALE = 200
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GRAY = (128, 128, 128)

        tip_x = self.POS[0]+self.len_wheel*sin(self.theta_rod)*SCALE
        tip_y = self.POS[1]-self.len_wheel*cos(self.theta_rod)*SCALE
        POSTIP = np.array([tip_x, tip_y])
        POSWHEEL = np.array(([tip_x+self.rad_out*sin(self.theta_wheel)*SCALE/4, tip_y-self.rad_out*cos(self.theta_wheel)*SCALE/4]))
        #print(POSTIP)
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, self.POS, POSTIP, 10)
        pygame.draw.circle(self.screen, GRAY, POSTIP, self.rad_out*2*SCALE/4)
        pygame.draw.circle(self.screen, RED, POSWHEEL, self.rad_out*2*SCALE/5/4)
        img = self.hint_font.render("torque  : % .4f" %self.torque, True, BLACK)
        #img2 = self.hint_font.render("voltage: % .4f" %self.voltage, True, BLACK)
        img3 = self.hint_font.render("Evaluation Run %d" %eval_run, True, BLACK)
        self.screen.blit(img, (self.origin_x, self.origin_y/2-50))
        #self.screen.blit(img2, (self.origin_x, self.origin_y/2-30))
        self.screen.blit(img3, (self.origin_x/5, self.origin_y/2-50))

        pygame.display.update()


    def step(self, action):
        q1, q1_dot, q2_dot = self.state

        #q1 = self.theta_rod
        q2 = self.theta_wheel
        #q1_dot = self.theta_rod_dot
        #q2_dot = self.theta_wheel_dot
        #l1 = self.len_rod
        #l2 = self.len_wheel
        #m1 = self.mass_rod
        #m2 = self.mass_wheel
        #I1 = self.momentum_rod
        I2 = self.momentum_wheel
        dt = self.dt
        #g = self.gravity
        #action_scale = self.max_torque

        #torque = action * action_scale
        #torque = np.clip(torque, -self.max_torque, self.max_torque)

        if action == 0:
            torque = 0
        elif action == 1:
            torque = -self.max_torque
        elif action == 2:
            torque = self.max_torque
        else:
            torque = 1000000

        #Ip = m2*l2**2+I1+I2 + 2*l1*m1*l1**2+(1-2*l1)*m1*(0.5*(1-2*l1)+2*l1)**2
        #a = (m1*l1+m2*l2)*g*sin(angle_normalize(q1))
        Ip = self.Ip
        a = self.mbarg*sin(angle_normalize(q1))

        q1_dot = q1_dot + ((a-torque)/(Ip-I2))*dt
        q1 = angle_normalize(angle_normalize(q1) + q1_dot * dt)

        q2_dot = q2_dot + ((torque*Ip-a*I2)/I2/(Ip-I2))*dt
        q2_dot = np.clip(q2_dot, -self.wheel_max_speed, self.wheel_max_speed)
        q2 = angle_normalize(angle_normalize(q2) + q2_dot * dt)

        #state = np.array([q1[0], q1_dot[0], q2_dot[0]], dtype=np.float32)
        self.state = (q1, q1_dot, q2_dot)

        done = bool(
            q1 < -self.max_q1
            or q1 > self.max_q1
            or q1_dot < -self.max_q1dot
            or q1_dot > self.max_q1dot
        )

        self.theta_rod = q1
        self.theta_wheel = q2
        self.theta_rod_dot = q1_dot
        self.theta_wheel_dot = q2_dot
        self.torque = torque

        #costs = 1000 * angle_normalize(q1) ** 2 + 0.1 * q1_dot ** 2 + 0.001 * torque ** 2
        # costs = 1000 * angle_normalize(q1) ** 2 + 0.1 * q1_dot ** 2 + 0.001 * torque ** 2 + 0.00001 * q2_dot**2
        # costs = 100 * angle_normalize(q1) ** 2 + 0.00001 * q2_dot ** 2
        costs = 100 * angle_normalize(q1) ** 2 + 1 * q1_dot ** 2 + 100 * torque ** 2 + 0.001 * q2_dot ** 2


        #if abs(angle_normalize(q1)) < 0.001 and abs(q1_dot) < 0.001 and abs(q2_dot) < 0.1 :
        #    costs -= 1000
        #elif abs(angle_normalize(q1)) < 0.001 and abs(q1_dot) < 0.001 and abs(q2_dot) < 0.01 :
        #    costs -= 1000

        #return state, -costs, False, {}
        return np.array(self.state, dtype=np.float32), -costs, done, {}

    def close(self):
        pygame.display.quit()
        pygame.quit()


def angle_normalize(th):
    return ((th+pi)%(2*pi))-pi