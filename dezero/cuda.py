#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
from dezero import Variable


# In[3]:


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data
        
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


# In[4]:


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data
        
    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


# In[5]:


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data
        
    if not gpu_enable:
        raise Exception("CuPy cannot be loaded. Install CuPy!")
    return cp.asarray(x)


# In[ ]:




