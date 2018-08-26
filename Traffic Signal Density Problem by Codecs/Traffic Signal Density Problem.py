
# coding: utf-8

# In[6]:


import sys
import os
import numpy
import pandas
import matplotlib
import seaborn


# In[7]:


print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('pandas: {}'.format(pandas.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[65]:


da=pd.read_csv('signnames.csv')


# In[66]:


print(da.columns)


# In[67]:


print(da.shape)


# In[68]:


print(da.describe())


# In[69]:


print(data6)


# In[70]:


import pickle as pickle


# In[71]:


da.hist(figsize = (20,20))
plt.show()


# In[75]:


def load_pickled_data(file,columns):
    with open(file,mode='rb') as f:
        dataset = pickle.load(f)
        return tuple(map(lambda c:data[c],columns))


# In[76]:


train_dataset_file = 'traffic-signs-data/train.p'
test_dataset_file = 'traffic-signs-data/test.p'


# In[83]:


X_train,y_train = load_pickled_data(train_dataset_file,['features','labels'])
X_test,y_test = load_pickled_data(test_dataset_file,['features','labels'])
n_train = y_train.shape[0]
n_test = y_test.shape[0]
image_shape = x_train[0].shape
n_classes=len(set(y_train))
print("number of training examples =",n_train)


# In[12]:


data = np.random.rand(20,18)
da=sns.heatmap(data)

