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


class Pendulum:
    def __init__(self, rend, seed):
        self.np_random = np.random.seed(seed)
        #for i in range(10):
        #    print(np.random.uniform(low=-3.5, high=3.5)*pi/180)

        self.theta_rod = 0
        self.theta_wheel = 0
        self.theta_rod_dot = 0
        self.theta_wheel_dot = 0
        self.len_rod = 0.35
        self.len_wheel = 0.5
        self.mass_rod = 20
        self.rad_out = 0.055
        self.rad_in = 0.032
        self.t = 0.02
        self.rho = 7870
        self.mass_wheel = (self.rad_out**2-self.rad_in**2)*pi*self.t*self.rho
        self.momentum_rod = self.mass_rod*(2*self.len_rod)**2/33
        self.momentum_wheel = self.mass_wheel*(self.rad_out**2+self.rad_in**2)/2
        self.dt = 0.001
        self.gravity = 9.81
        self.wheel_max_speed = 20
        self.max_torque = 10
        self.torque = 0
        self.voltage = 0

        self.action_space = 100

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


    def reset(self, saved):

        roll_range = -3 #in degree
        self.ang = roll_range
        #reset_max_speed = 3

        if saved == None:
            self.theta_rod = np.random.uniform(low=-roll_range * pi / 180, high=roll_range * pi / 180)
        elif saved != None:
            self.theta_rod = roll_range * 1 * pi / 180
        #print(self.theta_rod*180/pi)

        #self.theta_rod = np.random.uniform(low=-roll_range*pi/180, high=roll_range*pi/180)
        #self.theta_rod = np.random.uniform(low=0, high=roll_range*pi/180)
        #self.theta_rod = roll_range*0.3*pi/180
        #print(self.theta_rod)

        #print("\n",self.theta_rod)
        #self.theta_rod = (np.random.random()*2-1)*roll_range*pi/180
        #self.theta_wheel = 0
        self.theta_rod_dot = 0
        #self.theta_rod_dot = (np.random.random() * 2 - 1) * reset_max_speed
        self.theta_wheel_dot = 0

        '''
        ##############################
        # reset wheel velocity
        wheel_vel = 20
        if saved == None:
            self.theta_wheel_dot = np.random.uniform(low=-wheel_vel, high=wheel_vel)
        else:
            self.theta_wheel_dot = wheel_vel

        self.theta_rod = 0
        self.theta_rod_dot = 0

        ##############################
        '''

        state = np.array([self.theta_rod, self.theta_rod_dot, self.theta_wheel_dot], dtype=np.float32)
        return state


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
        q1 = self.theta_rod
        q2 = self.theta_wheel
        q1_dot = self.theta_rod_dot
        q2_dot = self.theta_wheel_dot
        l1 = self.len_rod
        l2 = self.len_wheel
        m1 = self.mass_rod
        m2 = self.mass_wheel
        I1 = self.momentum_rod
        I2 = self.momentum_wheel
        dt = self.dt
        g = self.gravity
        #gear_ratio = 25
        #kt = 0.0229
        #ke = 0.0229
        #R = 0.71
        #action_scale = 12
        #gear_ratio = 7
        #kt = 0.55
        #ke = 0.34
        #R = 0.38
        #action_scale = 48

        action_scale = self.max_torque

        torque = action * action_scale
        #voltage = action * action_scale
        #torque = gear_ratio*kt/R*(voltage-ke*gear_ratio*q2_dot)
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        #print("v",voltage)
        # print("T",torque)
        #print("q2dot",q2_dot)
        #print("q2", q2)

        Ip = m1*l1**2+m2*l2**2+I1+I2
        a = (m1*l1+m2*l2)*g*sin(angle_normalize(q1))

        newq1_dot = q1_dot + ((a-torque)/(Ip-I2))*dt
        #print("rod ang_vel",newq1_dot)
        #newq1_dot = np.clip(newq1_dot, -self.wheel_max_speed, self.wheel_max_speed)
        #print("rod ang_vel",newq1_dot)
        newq1 = angle_normalize(angle_normalize(q1) + newq1_dot * dt)
        #print("rod angle",newq1)

        newq2_dot = q2_dot + ((torque*Ip-a*I2)/I2/(Ip-I2))*dt
        newq2_dot = np.clip(newq2_dot, -self.wheel_max_speed, self.wheel_max_speed)
        #print("wheel ang_vel",newq2_dot)
        newq2 = angle_normalize(angle_normalize(q2) + newq2_dot * dt)
        #print("wheel angle",newq2)

        #print("torque",torque)
        #print("\n")
        #print([torque, newq1[0], newq2[0], newq1_dot[0], newq2_dot[0]])

        # print("newq1",newq1)

        state = np.array([newq1[0], newq1_dot[0], newq2_dot[0]], dtype=np.float32)

        self.theta_rod = newq1
        self.theta_wheel = newq2
        self.theta_rod_dot = newq1_dot
        self.theta_wheel_dot = newq2_dot
        self.torque = torque
        #self.voltage = voltage

        #print(newq2_dot)
        '''
        if angle_normalize(q1) < self.ang*pi/180/10:
            costs = 1000 * angle_normalize(q1)**2 + 0.1 * q1_dot**2 + 0.001 * voltage**2
        elif angle_normalize(q1) >= self.ang*pi/180/10:
            costs = 1000 * angle_normalize(q1) ** 2 + 0.001*48**2
        '''

        #costs = 1000 * angle_normalize(q1) ** 2 + 0.1 * q1_dot ** 2 + 0.001 * torque ** 2
        # costs = 1000 * angle_normalize(q1) ** 2 + 0.1 * q1_dot ** 2 + 0.001 * torque ** 2 + 0.00001 * q2_dot**2
        # costs = 100 * angle_normalize(q1) ** 2 + 0.00001 * q2_dot ** 2
        costs = 100 * angle_normalize(q1) ** 2 + 1 * q1_dot ** 2 + 100 * torque ** 2 + 0.001 * q2_dot ** 2


        #if abs(angle_normalize(q1)) < 0.001 and abs(q1_dot) < 0.001 and abs(q2_dot) < 0.1 :
        #    costs -= 1000
        #elif abs(angle_normalize(q1)) < 0.001 and abs(q1_dot) < 0.001 and abs(q2_dot) < 0.01 :
        #    costs -= 1000

        return state, -costs, False, {}

    '''
    def step_q2dot(self, action):
        q1 = self.theta_rod
        q2 = self.theta_wheel
        q1_dot = self.theta_rod_dot
        q2_dot = self.theta_wheel_dot
        l1 = self.len_rod
        l2 = self.len_wheel
        m1 = self.mass_rod
        m2 = self.mass_wheel
        I1 = self.momentum_rod
        I2 = self.momentum_wheel
        dt = self.dt
        g = self.gravity

        action_scale = self.max_torque

        torque = action * action_scale
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        Ip = m1*l1**2+m2*l2**2+I1+I2
        a = (m1*l1+m2*l2)*g*sin(angle_normalize(q1))

        newq1_dot = q1_dot + ((a-torque)/(Ip-I2))*dt
        newq1 = angle_normalize(angle_normalize(q1) + newq1_dot * dt)

        newq2_dot = q2_dot + ((torque*Ip-a*I2)/I2/(Ip-I2))*dt
        newq2_dot = np.clip(newq2_dot, -self.wheel_max_speed, self.wheel_max_speed)
        newq2 = angle_normalize(angle_normalize(q2) + newq2_dot * dt)

        state = np.array([newq1[0], newq1_dot[0], newq2_dot[0]], dtype=np.float32)

        self.theta_rod = newq1
        self.theta_wheel = newq2
        self.theta_rod_dot = newq1_dot
        self.theta_wheel_dot = newq2_dot
        self.torque = torque

        #costs = 1000 * angle_normalize(q1) ** 2 + 0.1 * q1_dot ** 2 + 0.001 * torque ** 2
        #costs = 1000 * angle_normalize(q1) ** 2 + 0.1 * q1_dot ** 2 + 0.001 * torque ** 2 + 0.00001 * q2_dot**2
        # costs = 100 * angle_normalize(q1) ** 2 + 1 * q1_dot ** 2
        costs = 0.0001 * q2_dot ** 2

        
        #if abs(angle_normalize(q1)) < 0.002 and abs(q1_dot) < 0.002 and abs(q2_dot) < 0.1 :
        #    costs -= 100
        #elif abs(angle_normalize(q1)) < 0.001 and abs(q1_dot) < 0.001 and abs(q2_dot) < 0.01 :
        #    costs -= 1000
        
        return state, -costs, False, {}
    '''

    def close(self):
        pygame.display.quit()
        pygame.quit()


def angle_normalize(th):
    return ((th+pi)%(2*pi))-pi