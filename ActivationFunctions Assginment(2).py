#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
  
x = np.arange(-10, 11, 1)
print(x)


# In[2]:


def plot_graph(y,ylabel):
    plt.figure()
    plt.plot(x,y, 'o--')
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel(ylabel)
    plt.show()


# In[3]:


y = list(map(lambda n: 1 if n>0.5 else 0, x))
plot_graph(y,"STEP(X)")


# In[4]:


y = 1 / (1 + np.exp(-x))
plot_graph(y, "Sigmoid(X)")


# In[5]:


y = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
plot_graph(y, "TanH(X)")


# In[6]:


y = list(map(lambda a: a if a>=0 else 0, x))
plot_graph(y,"ReLU(X)")


# In[24]:


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
y = elu(x)
plot_graph(y,"ELU(X)")


# In[34]:


def selu(x, lambdaa = 1.0507, alpha = 1.6732):
    return np.where(x<0, lambdaa*alpha*(np.exp(x)-1),x)
y = selu(x)
plot_graph(y,"SELU(X)")


# In[ ]:




