import json
from simulation import Robot_manipulator 
from online_trainer import OnlineTrainer
import math
from backpropagation import NN
import numpy as np
import matplotlib.pyplot as plt
robot=Robot_manipulator()
robot.set_theta(np.pi,np.pi)


HL_size= 10 # nbre neurons of Hiden layer
network = NN(2, HL_size, 2)    
target=[.7,0.2]
trainer1 = OnlineTrainer(robot,network)

show_animation = True



print('New target :',(target[0], target[1]))
thetas1=[]
thetas2=[]

def main()  :
    choice = input('Do you want to load previous network? (y/n) --> ')
    if choice == 'y':
        with open('last_w.json') as fp:
            json_obj = json.load(fp)
        for i in range(2):
            for j in range(HL_size):
                network.wi[i][j] = json_obj["input_weights"][i][j]
        for i in range(HL_size):
            for j in range(2):
                network.wo[i][j] = json_obj["output_weights"][i][j]
    choice = ''

    choice = input('Do you want to learn? (y/n) --> ')
    
    
    while (choice == 'y'):
        thetas1,thetas2 = trainer1.train(target)
        trainer = trainer1
        print("starting the training")
        trainer.training = True

        ax = robot.plot_arm(target[0], target[1])
        #line1, line2, pt1 = robot.draw_robot()
  
        #name=input('name of the gif (ajoutez l extension) --> ')
        #robot.train(thetas1,thetas2,line1,line2,pt1,ax,name)
        
    

if __name__ == "__main__":
    main()
    json_obj = {"input_weights": network.wi, "output_weights": network.wo}
    with open('last_w.json', 'w') as fp:
        json.dump(json_obj, fp)

    print("The last weights have been stored in last_w.json")
