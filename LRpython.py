#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Importing the dataset
import pandas as pd


# In[7]:
import sys

if len(sys.argv) > 1:
    input_file  = sys.argv[1]
else:
    print("Please provide an input file")
    sys.exit(-1)
    
df = pd.read_csv(input_file)
print(df.head())



# In[8]:


df


# In[9]:


df['y'].head()


# In[10]:


df[' x'].head()


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[13]:


plt.scatter(df['y'], df[' x'])
plt.show()
plt.savefig("py_orig.png")


# In[14]:


import numpy as np 
from sklearn.linear_model import LinearRegression


# In[15]:


x = np.array(df[' x']).reshape((-1, 1))
y = np.array(df['y'])


# In[16]:


model = LinearRegression()
model.fit( x, y)


# In[17]:


intercept = model.intercept_
slope = model.coef_
r_sq = model.score( x,y)


# In[18]:


print(f"intercept: {intercept}")
print(f"slope: {slope}")
print(f"r sqared: {r_sq}")


# In[19]:


y_pred = model.predict( x)
y_pred


# In[20]:


plt.plot(df[' x'], y_pred)
plt.show()
plt.scatter(df[' x'], df['y'])
plt.plot(df[' x'], y_pred)
plt.show()

# In[21]:


plt.scatter(df[' x'], df['y'])
plt.plot(df[' x'], y_pred)
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
plt.savefig("py_lm.png")







