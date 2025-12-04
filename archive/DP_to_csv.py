# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:40:06 2020

@author: Mohammad Asif Zaman
Double pendulum motion animation using FuncAnimation()
"""

from __future__ import print_function   
from scipy.integrate import solve_ivp

import numpy as np
import pandas as pd


m1 = 2                 # mass of pendulum 1 (in kg)
m2 = 1                 # mass of pendulum 2 (in kg)
L1 = 1.4                 # length of pendulum 1 (in meter)
L2 = 1                 # length of pendulum 2 (in meter)
g = 9.8                # gravitatioanl acceleration constant (m/s^2)

tfinal = 10000.0       # Final time. Simulation time = 0 to tfinal.
Nt = 751
t = np.arange(0, tfinal, 1/30)
tt = [0,tfinal]

# Differential equations describing the system
def double_pendulum(t,u,m1,m2,L1,L2,g):
    # du = derivatives
    # u = variables
    # p = parameters
    # t = time variable

    du = np.zeros(4)
    
    c = np.cos(u[0]-u[2])  # intermediate variables
    s = np.sin(u[0]-u[2])  # intermediate variables
    
    du[0] = u[1]   # d(theta 1)
    du[1] = ( m2*g*np.sin(u[2])*c - m2*s*(L1*c*u[1]**2 + L2*u[3]**2) - (m1+m2)*g*np.sin(u[0]) ) /( L1 *(m1+m2*s**2) )
    du[2] = u[3]   # d(theta 2)   
    du[3] = ((m1+m2)*(L1*u[1]**2*s - g*np.sin(u[2]) + g*np.sin(u[0])*c) + m2*L2*u[3]**2*s*c) / (L2 * (m1 + m2*s**2))
    
    return du


def output_csv(index):
    i0 = np.random.rand()*np.pi*2-np.pi # angle 1
    i1 = np.random.rand()*6-3 # angular velocity 1
    i2 = np.random.rand()*np.pi*2-np.pi # angle 2
    i3 = np.random.rand()*6-3 # angular velocity 2
    u0 = [i0, i1, i2, i3]    # initial conditions. 
    # u[0] = angle of the first pendulum
    # u[1] = angular velocity of the first pendulum
    # u[2] = angle of the second pendulum
    # u[3] = angular velocity of the second pendulum

    sol = solve_ivp(double_pendulum, tt, u0, args=(m1,m2,L1,L2,g), t_eval=t)

    u0 = sol.y[0,:]     # theta_1 
    u1 = sol.y[1,:]     # omega 1
    u2 = sol.y[2,:]     # theta_2 
    u3 = sol.y[3,:]     # omega_2 

    #raidians to sine and cosine
    v0 = np.sin(u0)
    v1 = np.cos(u0)
    v2 = np.sin(u2)
    v3 = np.cos(u2)

    # Creating csv
    df = pd.DataFrame({'Time_t': sol.t, 'sin(0_1)': v0, 'cos(0_1)': v1, 'ω_1': u1, 'sin(0_2)': v2, 'cos(0_2)': v3, 'ω_2': u3})
    df.to_csv("dpdatalong_csv"+str(index)+".csv",index=False)


for k in range(1):
    output_csv(k)