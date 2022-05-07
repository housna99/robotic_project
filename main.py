import math
import matplotlib.pyplot as plt
from regex import R

l1=2
l2=1.5

theta_start=0
theta_end=math.pi/2

n_theta=2
theta_1=[]
theta_2=[]

x0=0
y0=0

for i in range(0, n_theta ):
    theta_value=theta_start+i*(theta_end-theta_start)/(n_theta -1)
    theta_1.append(theta_value)
    theta_2.append(theta_value)
    print(theta_1)
    print(theta_2)

    k=1

    for i in theta_1:
        for j in theta_2:
            x1=l1*math.cos(i)
            y1=l1*math.sin(i)
            x2=x1+l2*math.cos(j)
            y2=x2+l2*math.sin(j)

            filename=str(k)+ '.png'
            k=k+1
            print(filename)

            plt.figure(1)
            plt.plot([x0,x1], [y0,y1], 'r')
            plt.plot([x1,x2], [y1,y2], 'b')
            plt.plot(x2,y2,marker='D')
            plt.legend(['arm','manipulator'])
            plt.xlim([0,5])
            plt.ylim([0,5])
            # filename=str(k)+ '.png'
            # k=k+1
            plt.savefig(filename)
            plt.clf()
            