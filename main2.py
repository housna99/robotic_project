import matplotlib.pyplot as plt
import numpy as np
from backpropagation import NN
from simulation import Robot_manipulator 
from online_trainer import OnlineTrainer
import math

# Similation parameters
Kp = 15
dt = 0.01

# Link lengths
#L1 = l2 = 1

# Set initial goal position to the initial end-effector position
x = 2
y = 0
#IA

robot=Robot_manipulator()
robot.set_theta(np.pi,np.pi/4)
HL_size= 10 # nbre neurons of Hiden layer
network = NN(2, HL_size, 2)    
target=[0.5,0.5]
show_animation = True

if show_animation:
    plt.ion()

theta1=robot.get_theta()[0]
theta2=robot.get_theta()[1]
GOAL_TH=0.0
def two_joint_arm(GOAL_TH , theta1 , theta2 ):
    """
    Computes the inverse kinematics for a planar 2DOF arm
    When out of bounds, rewrite x and y with last correct values
    """
    global x, y
    #x_prev, y_prev = None, None
    while True:
        # try:
        #     # if x is not None and y is not None:
        #     #     x_prev = x
        #     #     y_prev = y
        #     if np.sqrt(x**2 + y**2) > (robot.L1 + robot.L2):
        #         theta2_goal = 0
        #     else:
        #         theta2_goal = np.arccos(
        #             (x**2 + y**2 - robot.L1**2 - robot.L2**2) / (2 * robot.L1  * robot.L2))
        #     tmp = np.math.atan2(robot.L2 * np.sin(theta2_goal),
        #                         (robot.L1 + robot.L2 * np.cos(theta2_goal)))
        #     theta1_goal = np.math.atan2(y, x) - tmp

        #     if theta1_goal < 0:
        #         theta2_goal = -theta2_goal
        #         tmp = np.math.atan2(robot.L2 * np.sin(theta2_goal),
        #                             (robot.L1 + robot.L2 * np.cos(theta2_goal)))
        #         theta1_goal = np.math.atan2(y, x) - tmp

        #     theta1 = theta1 + Kp * ang_diff(theta1_goal, theta1) * dt
        #     theta2 = theta2 + Kp * ang_diff(theta2_goal, theta2) * dt
        # except ValueError as e:
        #     print("Unreachable goal"+e)
        # # except TypeError:
        # #     x = x_prev
        # #     y = y_prev
        wrist = plot_arm(theta1, theta2, x, y)

        # # check goal
        # d2goal = None
        # if x is not None and y is not None:
        #     d2goal = np.hypot(wrist[0] - x, wrist[1] - y)

        # if abs(d2goal) < GOAL_TH and x is not None:
        #     return theta1, theta2


def plot_arm(theta1, theta2, target_x, target_y):  # pragma: no cover
    shoulder = np.array([0, 0])
    for i in range(len(theta1)):
        elbow = shoulder + np.array([robot.L1  * np.cos(theta1[i]), robot.L1  * np.sin(theta1[i])])
        wrist = elbow + \
            np.array([robot.L2 * np.cos(theta1[i] + theta2[i]), robot.L2 * np.sin(theta1[i] + theta2[i])])

    if show_animation:
        plt.cla()

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')

        plt.plot([wrist[0], target_x], [wrist[1], target_y], 'g--')
        plt.plot(target_x, target_y, 'g*')

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        plt.show()
        plt.pause(dt)

    return wrist


def ang_diff(theta1, theta2):
    # Returns the difference between two angles in the range -pi to +pi
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


def click(event):  # pragma: no cover
    global x, y
    x = event.xdata
    y = event.ydata


def animation():
    from random import random
    global x, y
    trainer1 = OnlineTrainer(robot,network)
    
    for i in range(5):
        x = 2.0 * random() - 1.0
        y = 2.0 * random() - 1.0
        theta1, theta2 =  trainer1.train(target)


def main():  # pragma: no cover
    fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", click)
    # for stopping simulation with the esc key.
    fig.canvas.mpl_connect('key_release_event', lambda event: [
                           exit(0) if event.key == 'escape' else None])
    
    target=[0.5,-1]
    trainer1 = OnlineTrainer(robot,network)
    thetas1=[]
    thetas2=[]
    thetas1,thetas2 = trainer1.train(target)
   
    
    Fig, ax = robot.draw_env(target)
    line1, line2, pt1 = robot.draw_robot(Fig,ax)
    name='gif.gif'
    robot.train(thetas1,thetas2,line1,line2,pt1,Fig,name)

if __name__ == "__main__":
    # animation()
    main()
    