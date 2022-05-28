import matplotlib.pyplot as plt
import numpy as np
import math
from backpropagation import NN
from simulation import Robot_manipulator
from online_trainer import OnlineTrainer

# Similation parameters
dt = 0.1

# Link lengths
l1 =  1
l2 = 0.5

# Set initial goal position to the initial end-effector position
target_x = float(input('target_x:'))
target_y = float(input('target_y:'))
goal = True
target=[target_x,target_y]

# Check if the goal is reachable
if math.sqrt(target_x**2 + target_y**2) > (l1 + l2):
    print("Unreachable goal")
    goal = False

show_animation = True

if show_animation:
    plt.ion()

# Robot parameters
robot=Robot_manipulator(l1,l2)
robot.set_theta(np.pi,np.pi/2)
theta1=robot.get_theta()[0]
theta2=robot.get_theta()[1]

# Neural Network parameters
HL_size= 10 # nbre neurons of Hiden layer
network = NN(2, HL_size, 2) 
trainer1 = OnlineTrainer(robot,network)
trainer1.training = True

# Set of function allowing the robot to move to the goal position
def two_joint_arm():
       
        thetas1,thetas2 = trainer1.train(target)
        for i in range(len(thetas1)):
            wrist = plot_arm(thetas1[i], thetas2[i], target_x, target_y)
        return thetas1, thetas2

def plot_arm(theta1, theta2, target_x, target_y):  # pragma: no cover
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
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

def animation():
    global target_x, target_y
    theta1, theta2 = two_joint_arm()

def click(event):  # pragma: no cover
    global target_x, target_y
    target_x = event.xdata
    target_yy = event.ydata


def main():  # pragma: no cover
    fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", click)
    # for stopping simulation with the esc key.
    fig.canvas.mpl_connect('key_release_event', lambda event: [
                           exit(0) if event.key == 'escape' else None])
    


if __name__ == "__main__" and goal == True:
    animation()
    main()