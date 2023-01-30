#!/usr/bin/env python
# coding: utf-8

# In[124]:


import numpy as np
import matplotlib.pyplot as plt
#A=np.arange(1,13,2)
#B=np.arange(1,23,2)
#C=np.arange(1,33,2)
#print(A)
#print(B)
#print(C)
x=np.arange(-np.pi,np.pi,0.1)
y=np.pi/2+2*(np.sin(x)+np.sin(3*x)/3+np.sin(5*x)/5+np.sin(7*x)/7+np.sin(9*x)/9+np.sin(11*x)/11)
y_phi=np.pi/2+2*(np.sin(x)+np.sin(3*x)/3+np.sin(5*x)/5+np.sin(7*x)/7+np.sin(9*x)/9+np.sin(11*x)/11+np.sin(13*x)/13+np.sin(15*x)/15+np.sin(17*x)/17+np.sin(19*x)/19+np.sin(21*x)/21)
y_phi_2=np.pi/2+2*(np.sin(x)+np.sin(3*x)/3+np.sin(5*x)/5+np.sin(7*x)/7+np.sin(9*x)/9+np.sin(11*x)/11+np.sin(13*x)/13+np.sin(15*x)/15+np.sin(17*x)/17+np.sin(19*x)/19+np.sin(21*x)/21+np.sin(23*x)/23+np.sin(25*x)/25+np.sin(27*x)/27+np.sin(29*x)/29+np.sin(31*x)/31)
plt.plot(x,y, color = 'green')
plt.plot(x,y_phi, color = 'red')
plt.plot(x,y_phi_2, color = 'purple')


# In[28]:


import matplotlib.pyplot as plt
import numpy as np
import math
arr = math.sin(i*math.pi)/i
for i in arr:
    arr=arr+i


# In[100]:


import numpy as np
import matplotlib.pylab as plt
x=np.arange(-math.pi,4*math.pi/3,math.pi/3)
A=np.arange(1,13,2)
print(A)
print(x)
#y=np.sin(np.dot(A,x))
print(y)
#z=sum(y)
#print(z)
#plt.plot(x,y)
#plt.show()


# In[37]:


1+9+25+49+81


# In[ ]:




