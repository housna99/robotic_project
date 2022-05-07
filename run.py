from simulation import Robot_manipulator 
import math
import numpy as np
import matplotlib.pyplot as plt
robot=Robot_manipulator()
robot.set_theta(np.pi/2,0)
print(robot.get_theta())
print(robot.get_bras_x())
print(robot.get_coord_bras1())
print(robot.get_coord_bras2())
print(robot.get_coord_pince())
# robot.generate_liste_of_coord() #mn le trainer
# Fig,ax=robot.draw_env([5,6])
# line1, line2, line3, pt1 = robot.draw_robot(Fig,ax)
# print(line1,line2,line3,pt1)
# robot.train(#les th de online_trainer# ,line1,line2,line3,pt1,Fig
# ) 
show_animation = True

if show_animation:
    plt.ion()
fig = plt.figure()


#fig.canvas.mpl_connect("button_press_event", click)
    # for stopping simulation with the esc key.
#fig.canvas.mpl_connect('key_release_event', lambda event: [
                           
#robot.two_joint_arm()
#robot.generate_liste_of_coord([1.5],[2])
# a=robot.draw_env([3.5,1])
# robot.draw_robot(a[0],a[1])
thetas1=[]
thetas2=[]
thetas1.append(robot.get_theta()[0])
thetas2.append(robot.get_theta()[1])
n_theta=50
theta_start=0
theta_end=math.pi
for i in range(0,n_theta):
    theta_value=theta_start+i*(theta_end-theta_start)/(n_theta -1)
    thetas1.append(theta_value)
    thetas2.append(theta_value)
    
target=[-0.755,1]
print(len(thetas1))
def animate() : 
    #global thetas1,thetas2,thetas3,target_x,target_y
    
    tar = [target[0],target[1]]
    Fig, ax = robot.draw_env(tar)
    
    line1, = ax.plot([0.,robot.L1], [0.,0.], 'o-b', lw=10 , markersize=20)
    line2, = ax.plot([robot.L1,robot.L1+robot.L2], [0.,0.], 'o-r', lw=10 , markersize=20)
            #line3, = ax.plot([self.L1+self.L2,self.L1+self.L2], [0.,0.], 'o-', lw=10 , markersize=20)        
    pt1    = ax.scatter([robot.L1+robot.L2],[0.],marker="$\in$",s=800,c="black",zorder=3)
    robot.train(thetas1,thetas2,line1,line2,pt1,Fig)    

animate()
