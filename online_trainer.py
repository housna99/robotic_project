import time
import math


class OnlineTrainer:
    def __init__(self, robot, NN):
        """
        Args:
            robot (Robot): a robot instance of robot manipulator simulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN
        self.running = True
        self.training = False

        self.alpha = [1,1] # en 2ddl #[1/6,1/6,1/(math.pi)]  # normalition avec limite du monde cartesien = -3m � + 3m

    def train(self, targett):
        
        position = self.robot.get_coord_pince()
        alpha_1 = self.alpha[0]
        alpha_2 = self.alpha[1]
        th1 = []
        th2 = []
             
        target = [float(targett[0]),float(targett[1])]
        #print(target)
        #network_input[0] = 
        #network_input[1] = 
        robot_a_bouge = time.time()
        i=0
        while abs(position[0]-target[0]) > 0.001 and  abs(position[1]-target[1]) > 0.001 : 
            debut = time.time()
            network_input = [(position[0]-target[0])*self.alpha[0], (position[1]-target[1])*self.alpha[1]]
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t  # Fonction à changer
            crit_av= alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*(position[1]-target[1])*(position[1]-target[1]) 
                                 
            diff = time.time() - robot_a_bouge
            thetas = self.robot.get_theta()
            theta_temp1 = thetas[0] + command[0] * diff
            theta_temp2 = thetas[1] + command[1] * diff
            self.robot.set_theta(theta_temp1,theta_temp2 )    
            robot_a_bouge = time.time()  
            th1.append(theta_temp1)
            th2.append(theta_temp2)
            #print('th1', th1)
            #print('len(th1): ',th1)
            time.sleep(0.050) # attend delta t

            position = self.robot.get_coord_pince() #  obtient nvlle pos robot instant t+1       # Fonction à changer             
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            #i+=1
            crit_ap= alpha_1*(position[0]-target[0])*(position[0]-target[0]) + alpha_2*(position[1]-target[1])*(position[1]-target[1]) 
            selfthetas1,selfthetas2 = self.robot.get_theta()

            if self.training:
                delta_t = (time.time()-debut)

                grad = [
                    2*(-1)*alpha_1*delta_t*(self.robot.L1*math.sin(selfthetas1)+self.robot.L2*math.sin(selfthetas1 + selfthetas2))*(target[0] - position[0])
                    -2*(-1)*alpha_2*delta_t*(self.robot.L1*math.cos(selfthetas1)+ self.robot.L2*math.cos(selfthetas1 + selfthetas2))*(target[1] - position[1]),
                    
                    -2*delta_t*(self.robot.L2*math.sin(selfthetas1 + selfthetas2))*(target[0] - position[0])+2*delta_t*(target[1] - position[1])*(self.robot.L2*math.cos(selfthetas1 + selfthetas2))

                                    
                ]
                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                
                if (crit_ap <= crit_av) :
                    self.network.backPropagate(grad, 0.5,0)# grad, pas d'app, moment
                else :
                    #self.network.random_update(0.001)
                    self.network.backPropagate(grad, 0.5 ,0)
                    
   
                
        return th1,th2 