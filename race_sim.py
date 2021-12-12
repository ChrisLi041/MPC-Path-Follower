import re
from sys import exec_prefix
import matplotlib.pyplot as plt
import numpy as np
from MPC_path_follower import KinMPCPathFollower
import ref_traj_interp as tj 
from scipy.signal import savgol_filter


# Run MPC
if __name__ == '__main__':

    # ---------------------------------------------------------------
    #Import refTraj-------------------------------------------------
    # ---------------------------------------------------------------

    filename = '/home/han98122/Repositories/C231A/inputs/traj_race_cl_2.csv' #Define the file path
    dt = 0.2
    data = tj.extractData(filename)
    refTraj = tj.interpTime(data,dt)

    # ---------------------------------------------------------------
    #pre-processing reference trajectory data------------------------
    # ---------------------------------------------------------------

    temp2 = refTraj[:,3] + 0.5 * np.pi
    temp2 = np.unwrap(temp2)
    print(temp2.shape)
    # temp2 = (temp2 + np.pi) % (2 * np.pi) - np.pi
    # refTraj[:,3] = temp2
    # istrue = temp2 == refTraj[:,3]
    refTraj[:,3] = np.unwrap(refTraj[:,3])  #psi angle from the reference profile is wrapped around (-pi,pi). Unwrapping is needed here

    # An offset of 90 degrees is needed
    refTraj[:,3] += 0.5 * np.pi

    #Due to noisy reference psi angles, it was filtered for better tracking performance (parameters for the sav-gol filter were manually tuned)
    refTraj[:,3] = savgol_filter(refTraj[:,3],21,5)



    #plot of the Reference Trajectory
    # plt.figure()
    # plt.plot(refTraj[:,1],refTraj[:,2])
    # plt.show()

    '''
    Plots of the reference Psi (steering angles) in degrees
    '''
    # plt.figure()
    # plt.plot(refTraj[:,0],refTraj[:,3]*180/np.pi)
    # plt.plot(refTraj[:,0],temp2*180/np.pi)
    # plt.xlabel('Time [s]')
    # plt.ylabel('$\Psi$ [rad]')
    # plt.legend(['raw','2'])
    # plt.show()

    '''
    Plot of the velocity profile
    '''
    # plt.figure()
    # plt.plot(refTraj[:,0],refTraj[:,4])
    # plt.xlabel('Time [s]')
    # plt.ylabel('$v$ [m/s]')
    # plt.show()

    '''
    Run MPC
    '''
    #Define the simulation space
    N = 10                  # N*dt = 1 second prediction horizon
    M = len(refTraj)-N      # Total simulation length (iterations with dt = 0.2 seconds)

    #Initialize states for the MPC
    z0 = np.zeros((4))

    #Manually assign a non-zero (close to the initial velocity from the reference velocity profile) value for the initial state to be used by the MPC
    # z0[3] = 1
    z_cl = z0
    z_curr = z0

    #Initialize the MPC controller
    kmpc = KinMPCPathFollower()

    for iter in range(M):

        print(iter)

        #Update Initial condition
        kmpc._update_initial_condition(z_curr[0], z_curr[1], z_curr[2], z_curr[3])
        
        # Update reference
        kmpc._update_reference(refTraj[iter:N+iter,1],
                refTraj[iter:N+iter,2], 
                refTraj[iter:N+iter,3], 
                refTraj[iter:N+iter,4])	

        # kmpc._add_cost()

        #Solve MPC(CFTOCP)
        sol_dict = kmpc.solve()

        #Update previous input
        u_temp = sol_dict['u_control'].flatten()            # u_control is the control input to apply based on solution
        kmpc._update_previous_input(u_temp[0], u_temp[1])

        #Update current state
        z_temp = sol_dict['z_mpc']
        z_curr = z_temp[1] #not z[0]

        #Store CL trajectory info
        z_cl = np.vstack((z_cl,z_curr.reshape(1,4)))


        # for key in sol_dict:
        #     print(key, sol_dict[key])



    """
    Simulation Results
    """
    #Calculate/estimate elapsed time (to finish the race)
    
    estDist = np.sqrt((z_cl[-1,0] - refTraj[-1,2])**2 + (z_cl[-1,1] - refTraj[-1,3])**2)
    estTime = estDist / z_cl[-1,3]
    print('--------------------------------------------------------')
    print(f'The estimated race completion time delay is : {estTime} [s]')
    print(f'The optimal race completion time was : {refTraj[-1,0]} [s]')
    print(f'The estimated total race completion time using MPC is : {estTime + refTraj[-1,0]} [s]')
    #Calculate error
    e_x = z_cl[:,0] - refTraj[0:M+1,1]
    e_y = z_cl[:,1] - refTraj[0:M+1,2]
    e_v = z_cl[:,3] - refTraj[0:M+1,4]
    e_psi = z_cl[:,2] - refTraj[0:M+1,3] 

    #Plot closedloop trajectory
    plt.figure()
    plt.plot(z_cl[:,0],z_cl[:,1],'-r')
    plt.plot(refTraj[:,1],refTraj[:,2],'--k')
    plt.legend(['CL Trajectory','Reference Trajectory'])
    plt.show()

    #Plot errors
    plt.figure(figsize = (30,5))
    plt.subplot(1,4,1)
    plt.plot(refTraj[0:M+1,0],e_x)
    plt.xlabel('Time [s]')
    plt.ylabel('e_x [m]')

    plt.subplot(1,4,2)
    plt.plot(refTraj[0:M+1,0],e_y)
    plt.xlabel('Time [s]')
    plt.ylabel('e_y [m]')

    plt.subplot(1,4,3)
    plt.plot(refTraj[0:M+1,0],e_psi)
    plt.xlabel('Time [s]')
    plt.ylabel('$e_{\Psi}$ [rad]')

    plt.subplot(1,4,4)
    plt.plot(refTraj[0:M+1,0],e_v)
    plt.xlabel('Time [s]')
    plt.ylabel('e_v [m/s]')   
    plt.show()     

    #Plot psi 
    plt.figure()
    plt.plot(refTraj[0:M+1,0],z_cl[:,2],'-r')
    plt.plot(refTraj[0:M+1,0] ,refTraj[0:M+1,3],'--k')
    plt.legend(['Closed-loop','Reference'])
    plt.xlabel('Time [s]')
    plt.ylabel('$\Psi$ [rad]')
    plt.show()

    #Plot psi 
    plt.figure()
    plt.plot(refTraj[0:M+1,0],z_cl[:,3],'-r')
    plt.plot(refTraj[0:M+1,0] ,refTraj[0:M+1,4],'--k')
    plt.legend(['Closed-loop','Reference'])
    plt.xlabel('Time [s]')
    plt.ylabel('$v$ [m/s]')
    plt.show()

    #Plot x
    plt.figure()
    plt.plot(refTraj[0:M+1,0],z_cl[:,0],'-r')
    plt.plot(refTraj[0:M+1,0] ,refTraj[0:M+1,1],'--k')
    plt.legend(['Closed-loop','Reference'])
    plt.xlabel('Time [s]')
    plt.ylabel('$x$ [m]')
    plt.show()

    #Plot y
    plt.figure()
    plt.plot(refTraj[0:M+1,0],z_cl[:,1],'-r')
    plt.plot(refTraj[0:M+1,0] ,refTraj[0:M+1,2],'--k')
    plt.legend(['Closed-loop','Reference'])
    plt.xlabel('Time [s]')
    plt.ylabel('$y$ [m]')
    plt.show()