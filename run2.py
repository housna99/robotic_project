import json
from simulation import Robot_manipulator 
from online_trainer import OnlineTrainer
import math
from backpropagation import NN
import numpy as np
import matplotlib.pyplot as plt
robot=Robot_manipulator()
robot.set_theta(np.pi,np.pi/4)
HL_size= 10 # nbre neurons of Hiden layer
network = NN(2, HL_size, 2)    
target=[0.75,-0.2]
print('target:',target[0],target[1])
trainer1 = OnlineTrainer(robot,network)
thetas1=[]
thetas2=[]
thetas1,thetas2 = trainer1.train(target)
print('thethas1 :' ,thetas1)
print(thetas2)
Fig, ax = robot.draw_env(target)
line1, line2, pt1 = robot.draw_robot(Fig,ax)
robot.train(thetas1,thetas2,line1,line2,pt1,Fig)  
