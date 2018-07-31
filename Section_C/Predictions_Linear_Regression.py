
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


# In[2]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[3]:


filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


# In[5]:


test_df = pd.read_csv('yds_test2018.csv')
#test_df.drop('S_No',axis=1,inplace=True)
test_df


# In[6]:


country_map = {
                'Argentina':0,
                'Belgium':1,
                'Columbia':2,
                'Denmark':3,
                'England':4,
                'Finland':5 }


# In[7]:


test_df.Country = test_df.Country.map(country_map)
test_df


# In[12]:


X_test = test_df.drop(['Sales','S_No'],axis=1)
X_test.shape


# In[13]:


predictions = loaded_model.predict(X_test)
predictions


# In[14]:


test_df['Sales']=predictions
test_df


# In[15]:


Country_unmap = {
                    0:'Argentina',
                    1:'Belgium',
                    2:'Columbia',
                    3:'Denmark',
                    4:'England',
                    5:'Finland' }


# In[16]:


test_df.Country = test_df.Country.map(Country_unmap)
test_df


# In[17]:


test_df.to_csv('yds_submission2018.csv')

