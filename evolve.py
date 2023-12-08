#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


def evolve(y0: np.ndarray,t0:float,dt:float,n:int,f:callable,method:callable,scale=None,param=None)->np.ndarray:

    dof =len(y0)
    t= t0
    a = []
    y1=y0.copy()
    if scale is None:
        if param is None:
            for i in range(1,n):
                a.append(y1)
                y1 = method(y1,t,dt,f)
                t+=dt
        else:
            for i in range(1,n):
                a.append(y1)
                y1 = method(y1,t,dt,f,param)
                t+=dt

    else:
        if len(scale) != dof+1:
            raise Exception(f'scale vec must have dim={dof+1}')
        t_scale = scale[0]
        dt = dt/t_scale
        scal = np.array(scale[1:])
        y = y1/scal
        if param is None:
            for i in range(1,n):
                a.append(y1)
                y=method(y,t,dt,f)
                y1 = y*scal
                t+=dt
        else:
            for i in range(1,n):
                a.append(y1)
                y = method(y,t,dt,f,param)
                y1 = y*scal
                t+=dt

    return np.array(a)


# In[6]:


def euler_step(y: np.ndarray,t:float,dt:float,f:callable)->np.ndarray:

    y = y +dt*f(y,t)
    return y

def euler_step_param(y: np.ndarray,t:float,dt:float,f:callable,param)->np.ndarray:
    y = y +dt*f(y,t,param)

    return y

def rk4_step(y:np.ndarray,t:float,dt:float,f:callable)->np.ndarray:

    k1 = f(y,t)
    k2 = f(y+k1*dt/2.0,t+dt/2)
    k3 = f(y+k2*dt/2.0,t+dt/2.0)
    k4 = f(y+k3*dt,t+dt)
    k = dt *(k1+2*k2+2*k3+k4)/6.0
    y=y+k
    return y


def rk4_step_param(y:np.ndarray,t:float,dt:float,f:callable,param)->np.ndarray:
    k1 = f(y,t,param)
    k2 = f(y+k1*dt/2.0,t+dt/2.0,param)
    k3 = f(y+k2*dt/2.0,t+dt/2.0,param)
    k4 = f(y+k3*dt,t+dt,param)
    k = dt *(k1+2*k2+2*k3+k4)/6.0
    y=y+k
    return y

# In[ ]:




