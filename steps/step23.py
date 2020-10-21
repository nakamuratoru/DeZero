#!/usr/bin/env python
# coding: utf-8

# In[3]:


if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))    


# In[4]:


import numpy as np
from dezero import Variable


# In[5]:


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)


# In[ ]:




