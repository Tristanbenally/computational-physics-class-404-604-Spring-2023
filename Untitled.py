#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Homework 1:
# Solve the diffusion equation u_t=D u_{xx} with D=1 in the interval x=[0,1] from t=0 to t=0.1. 
# The initial condition is u(x,0)=0.5*(cos(13x)+1), the boundary condition is u(0,t)=1 and u(1,t)=0.
# After solving the equation, please also change tend to a large value to see how the solution behave in the long run.


# In[18]:


# EqHeat.py: solves heat equation via finite differences, 3-D plot
 
from numpy import *
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D 

Nx = 101;       tend=100000; Dt=5.2;  Dtout=100.; Nt = 3000;     Dx = 0.03;                                                            
KAPPA = 210.; SPH = 900.; RHO = 2700. # Conductivity, specf heat, density                                                      
T = zeros((Nx,2),float);  Tp = zeros((Nx,int(tend/Dtout)+1),float)  
# T[i,0] is the temperature at position i and old time, T[i, 1] is at the new time                                    

for ix in range (1, Nx - 1):  T[ix, 0] = 1/2*cos((13*ix)+1);               # Initial T
T[0,0] = 1.0 ;   T[0,1] = 0.                           # 1st & last T = 0
T[Nx-1,0] = 0. ; T[Nx-1,1] = 0.0
cons = KAPPA/(SPH*RHO)*Dt/(Dx*Dx);                             # constant=D*dt/Dx^2
m = 1                                                           # counter

toutn=0
t=0.
while t < tend:                                  
    for ix in range (1, Nx - 1):                       
        T[ix, 1] = T[ix, 0]+cons*(T[ix+1,0]-2*T[ix,0]+T[ix-1,0])# please finish this line using T[ix-1, 0],T[ix, 0],T[ix+1, 0] where T[ix, 0] means T at ix at the old step, T[ix, 1] means the new step                                                        
    t+=Dt
    for ix in range (1, Nx - 1):  T[ix, 0] = T[ix, 1] 

    if t >= toutn*Dtout: 
        p.plot(T)
        Tp[:,toutn] = T[:,1]   
        print(toutn)   
        toutn += 1                        

                
x = list(range(0, Nx))                       # Plot alternate pts
y = list(range(0, int(tend/Dtout)+1))                      
X, Y = p.meshgrid(x, y)                       

def functz(Tpl):                            
    z = Tpl[X, Y]       
    return z

Z = functz(Tp)              
fig = p.figure()                                          # Create figure
ax = Axes3D(fig)                                              
ax.plot_wireframe(X, Y, Z, color = 'r')                    
ax.set_xlabel('Position')                                     
ax.set_ylabel('time')
ax.set_zlabel('Temperature')
p.show()                               
print("finished") 


# In[20]:


# Burger's equation:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Model Parameters
xmin = -10.0   # left boundary
xmax = +10.0   # right boundary
Nx = 101      # number of grid points (including boundary)
tend = 10.0    # end time
dtout = 1.0   # time interval for outputs

# Set up the grid.
x = np.linspace(xmin, xmax, Nx)
dx = (xmax - xmin) / (Nx - 1)
dt = 0.8 * dx
U = np.zeros(Nx,)

# Give the initial profile.
t = 0.0
U = 0.2 + 0.8 * np.exp(-0.5 * x**2)

# Prepare for 3D outputs.
tp = [t]
Up = np.copy(U)

# Initiate the plot.
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
ax.plot(x, U, label=r"$t = {:.3G}$".format(t))
ax.set_xlabel(r"Position $x$")
ax.set_ylabel(r"$U(t,x)$")
ax.minorticks_on()

# Begin the simulation.
tout = t + dtout
while t < tend:
    # Backup the previous time step.
    Uold = np.copy(U)

    # Find the state at the next time step.
    if vel > 0:
        for ix in range(1, Nx - 1):
           U[ix]=Uold[ix]-(Uold[ix]*dt)*(Uold[ix]-U[ix-1])/dx # Task: implement upwind method here.
            
    else:
        for ix in range(1, Nx - 1):
           U[ix]=Uold[ix]-(Uold[ix]*dt)*(Uold[ix+1]-Uold[ix])/dx # Task: implement upwind method here.
            
    t += dt

        # Save the data after every dtout.
    if t >= tout:
        plt.plot(x, U, label=r"$t = {:.3G}$".format(t))
        tp.append(t)
        Up = np.vstack((Up, U))
        print("t = ", t)
        tout += dtout

ax.legend()

# Create 3D-view of the solution.
t, x = np.meshgrid(tp, x)
fig3D = plt.figure(figsize=(16,10))
ax3D = Axes3D(fig3D)
ax3D.plot_wireframe(t, x, Up.transpose(), color="red")
ax3D.set_xlabel("Time $t$")
ax3D.set_ylabel("Position $x$")
ax3D.set_zlabel(r"$U(t,x)$")

print("Done.")
plt.show()


# In[ ]:




