"""
Inverse kinematics of a two-joint arm
Left-click the plot to set the goal position of the end effector
Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)
Ref: P. I. Corke, "Robotics, Vision & Control", Springer 2017,
 ISBN 978-3-319-54413-7 p102
- [Robotics, Vision and Control]
(https://link.springer.com/book/10.1007/978-3-642-20144-8)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from sympy import true
from backpropagation import NN
from simulation import Robot_manipulator 
from online_trainer import OnlineTrainer
import math 

# Similation parameters
Kp = 15
dt = 0.1
robot=Robot_manipulator()
# Link lengths
l1=1
l2=0.5
robot.L1 =  l1
robot.L2 = l2

# Set initial goal position to the initial end-effector position
x = float(input('target_x:'))
y = float(input('target_y:'))
goal=True
if math.sqrt( x**2 + y**2) > (l1 + l2):
    print("Unreachable goal")
    goal = False
show_animation = True

if show_animation:
    plt.ion()


def two_joint_arm( theta1, theta2):
    while goal==True:
        thetas1,thetas2 = trainer1.train(target)
        for i in range(len(thetas1)):

            wrist = plot_arm(thetas1[i], thetas2[i], target_x, target_y)
        return thetas1, thetas2

robot=Robot_manipulator()
HL_size= 10 # nbre neurons of Hiden layer
network = NN(2, HL_size, 2) 
trainer1 = OnlineTrainer(robot,network)
robot.set_theta(np.pi,np.pi/2)
theta1=robot.get_theta()[0]
theta2=robot.get_theta()[1]
trainer1.training = True
target_x=x
target_y=y
target=[target_x,target_y]
show_animation = True

if show_animation:
    plt.ion()

theta1=robot.get_theta()[0]
theta2=robot.get_theta()[1]
def plot_arm(theta1, theta2, target_x, target_y):  # pragma: no cover
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    # wrist = elbow + \
    wrist=elbow+np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    if show_animation:
        plt.cla()

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')

        plt.plot([wrist[0], target_x], [wrist[1], target_y], 'g--')
        plt.plot(target_x, target_y, 'g*')

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        plt.show()
        plt.pause(dt)

    return wrist



def click(event):  # pragma: no cover
    global x, y
    x = event.xdata
    y = event.ydata


def animation():
    from random import random
    global x, y
    theta1=robot.get_theta()[0]
    theta2=robot.get_theta()[1]
    theta1, theta2 = two_joint_arm(
             theta1, theta2 )



def main():  # pragma: no cover
    choice = input('Do you want to load previous network? (y/n) --> ')
    if choice == 'y':
        with open('last_w2.json') as fp:
            json_obj = json.load(fp)
        for i in range(2):
            for j in range(HL_size):
                network.wi[i][j] = json_obj["input_weights"][i][j]
        for i in range(HL_size):
            for j in range(2):
                network.wo[i][j] = json_obj["output_weights"][i][j]
    choice = ''

    choice = input('Do you want to learn? (y/n) --> ')
    
    fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", click)
    # for stopping simulation with the esc key.
    fig.canvas.mpl_connect('key_release_event', lambda event: [
                           exit(0) if event.key == 'escape' else None])
    if (choice == 'y'):
        print("starting learning")
        animation()
        
if __name__ == "__main__" and goal==True:
    main()
    json_obj = {"input_weights": network.wi, "output_weights": network.wo}
    with open('last_w2.json', 'w') as fp:
        json.dump(json_obj, fp)

    print("The last weights have been stored in last_w.json")

