#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


missing_value_formats = ["Don't know/Not Sure/Refused/Missing", "Don´t know, refused or missing values", "Don't know/Not sure/Missing", "Don´t know/Not Sure/Refused/Missing", "Don't know/Not Sure Or Refused/Missing", "Don't know/Refused/Missing", "Missing", "Don´t Know/Not Sure", "Don't Know / Not Sure", "Don't Know/Not sure", "Don´t Know / Not sure", "Refused","Don't know/Not sure","Not asked or Missing","Don't know/Not Sure","Don't know / Not sure","Don't know / Not Sure", "Data do not meet the criteria for statistical reliability, data quality, or confidentiality (data are suppressed).", "D"]
df2 = pd.read_csv('DataSetResearchQuestion2.csv', na_values = missing_value_formats)


# In[3]:


df2.head()


# In[4]:


df2.shape


# In[5]:


df2["CVDSTRK3"].value_counts()


# In[6]:


df2["CVDSTRK3"].isnull().sum()


# In[7]:


df2=df2[~df2['CVDSTRK3'].isnull()]


# In[8]:


df2["CVDSTRK3"].isnull().sum()


# In[9]:


df2 = df2.rename(columns={'CVDSTRK3': 'Stroke'})


# In[10]:


df2.shape


# In[11]:


df2["SEQNO"].duplicated().value_counts()


# In[12]:


df2["SEQNO"].isnull().sum()


# In[13]:


df2 = df2.rename(columns={'SEQNO': 'RespondentID'})


# In[101]:


#df2["O_STATE"].value_counts()


# In[102]:


#df2["O_STATE"].isnull().sum()


# In[103]:


#df2['O_STATE'] = df2['O_STATE'].fillna(df2['O_STATE'].value_counts().index[0])
#df2["O_STATE"].value_counts()


# In[104]:


#df2["O_STATE"].isnull().sum()


# In[105]:


#df2 = df2.rename(columns={'O_STATE': 'State'})


# In[19]:


df2["MENTHLTH"].value_counts()


# In[20]:


df2["MENTHLTH"].isnull().sum()


# In[21]:


df2["MENTHLTH"].replace("Number of days", "Poor", inplace=True)
df2["MENTHLTH"].replace("None", "Good", inplace=True)
df2['MENTHLTH'] = df2['MENTHLTH'].fillna(df2['MENTHLTH'].value_counts().index[0])
df2["MENTHLTH"].value_counts()


# In[22]:


df2["MENTHLTH"].isnull().sum()


# In[23]:


df2 = df2.rename(columns={'MENTHLTH': 'MentalHealth'})


# In[24]:


df2["ADDEPEV2"].value_counts()


# In[25]:


df2["ADDEPEV2"].isnull().sum()


# In[26]:


df2['ADDEPEV2'] = df2['ADDEPEV2'].fillna(df2['ADDEPEV2'].value_counts().index[0])
df2["ADDEPEV2"].value_counts()


# In[27]:


df2["ADDEPEV2"].isnull().sum()


# In[28]:


df2 = df2.rename(columns={'ADDEPEV2': 'DepressiveDisorder'})


# In[29]:


df2["_RFHYPE5"].value_counts()


# In[30]:


df2["_RFHYPE5"].isnull().sum()


# In[31]:


df2['_RFHYPE5'] = df2['_RFHYPE5'].fillna(df2['_RFHYPE5'].value_counts().index[0])
df2["_RFHYPE5"].value_counts()


# In[32]:


df2["_RFHYPE5"].isnull().sum()


# In[33]:


df2 = df2.rename(columns={'_RFHYPE5': 'HighBloodPressure'})


# In[34]:


df2["TOLDHI2"].value_counts()


# In[35]:


df2["TOLDHI2"].isnull().sum()


# In[36]:


df2['TOLDHI2'] = df2['TOLDHI2'].fillna(df2['TOLDHI2'].value_counts().index[0])
df2["TOLDHI2"].value_counts()


# In[37]:


df2["TOLDHI2"].isnull().sum()


# In[38]:


df2 = df2.rename(columns={'TOLDHI2': 'HighBloodCholesterol'})


# In[39]:


df2["CVDINFR4"].value_counts()


# In[40]:


df2["CVDINFR4"].isnull().sum()


# In[41]:


df2['CVDINFR4'] = df2['CVDINFR4'].fillna(df2['CVDINFR4'].value_counts().index[0])
df2["CVDINFR4"].value_counts()


# In[42]:


df2["CVDINFR4"].isnull().sum()


# In[43]:


df2 = df2.rename(columns={'CVDINFR4': 'HeartAttack'})


# In[44]:


df2["_RFBMI5"].value_counts()


# In[45]:


df2["_RFBMI5"].isnull().sum()


# In[46]:


df2['_RFBMI5'] = df2['_RFBMI5'].fillna(df2['_RFBMI5'].value_counts().index[0])
df2["_RFBMI5"].value_counts()


# In[47]:


df2["_RFBMI5"].isnull().sum()


# In[48]:


df2 = df2.rename(columns={'_RFBMI5': 'Overweight/Obesity'})


# In[49]:


df2["SEX"].value_counts()


# In[50]:


df2["SEX"].isnull().sum()


# In[51]:


df2 = df2.rename(columns={'SEX': 'Sex'})


# In[52]:


df2["AGE"].value_counts()


# In[53]:


df2["AGE"].isnull().sum()


# In[54]:


df2["AGE"].replace("Age 65 or older", "65 or older", inplace=True)
df2["AGE"].replace("Age 55 - 64", "55 - 64", inplace=True)
df2["AGE"].replace("Age 45 - 54", "45 - 54", inplace=True)
df2["AGE"].replace("Age 35 - 44", "35 - 44", inplace=True)
df2["AGE"].replace("Age 25 - 34", "25 - 34", inplace=True)
df2["AGE"].replace("Age 18 - 24", "18 - 24", inplace=True)
df2['AGE'] = df2['AGE'].fillna(df2['AGE'].value_counts().index[0])
df2["AGE"].value_counts()


# In[55]:


df2["AGE"].isnull().sum()


# In[56]:


df2 = df2.rename(columns={'AGE': 'AgeRange'})


# In[57]:


df2["SMOKE100"].value_counts()


# In[58]:


df2["SMOKE100"].isnull().sum()


# In[59]:


df2['SMOKE100'] = df2['SMOKE100'].fillna(df2['SMOKE100'].value_counts().index[0])
df2["SMOKE100"].value_counts()


# In[60]:


df2["SMOKE100"].isnull().sum()


# In[61]:


df2 = df2.rename(columns={'SMOKE100': 'Smoke100'})


# In[62]:


df2["_RFBING5"].value_counts()


# In[63]:


df2["_RFBING5"].isnull().sum()


# In[64]:


df2['_RFBING5'] = df2['_RFBING5'].fillna(df2['_RFBING5'].value_counts().index[0])
df2["_RFBING5"].value_counts()


# In[65]:


df2["_RFBING5"].isnull().sum()


# In[66]:


df2 = df2.rename(columns={'_RFBING5': 'BingeDrinkers'})


# In[67]:


df2["EXERANY2"].value_counts()


# In[68]:


df2["EXERANY2"].isnull().sum()


# In[69]:


df2['EXERANY2'] = df2['EXERANY2'].fillna(df2['EXERANY2'].value_counts().index[0])
df2["EXERANY2"].value_counts()


# In[70]:


df2["EXERANY2"].isnull().sum()


# In[71]:


df2 = df2.rename(columns={'EXERANY2': 'PhysicalActivity'})


# In[72]:


df2["_FRTLT1"].value_counts()


# In[73]:


df2["_FRTLT1"].isnull().sum()


# In[74]:


df2["_FRTLT1"].replace("Consumed fruit one or more times per day", "Yes", inplace=True)
df2["_FRTLT1"].replace("Consumed fruit less than one time per day", "No", inplace=True)
df2['_FRTLT1'] = df2['_FRTLT1'].fillna(df2['_FRTLT1'].value_counts().index[0])
df2["_FRTLT1"].value_counts()


# In[75]:


df2["_FRTLT1"].isnull().sum()


# In[76]:


df2 = df2.rename(columns={'_FRTLT1': 'ConsumeFruitsEveryDay'})


# In[77]:


df2["_VEGLT1"].value_counts()


# In[78]:


df2["_VEGLT1"].replace("Consumed vegetables one or more times per day", "Yes", inplace=True)
df2["_VEGLT1"].replace("Consumed vegetables less than one time per day", "No", inplace=True)
df2['_VEGLT1'] = df2['_VEGLT1'].fillna(df2['_VEGLT1'].value_counts().index[0])
df2["_VEGLT1"].value_counts()


# In[79]:


df2["_VEGLT1"].isnull().sum()


# In[80]:


df2 = df2.rename(columns={'_VEGLT1': 'ConsumeVegetablesEveryDay'})


# In[81]:


df2.head()


# In[82]:


df2.shape


# In[83]:


# Exporting dataset:
from IPython.display import FileLink, FileLinks

df2.to_csv('ResearchQuestion2.csv', index=False)


# ## Relationship:

# In[99]:


data=df2.groupby(["AgeRange", "Sex", "Stroke"])["Stroke".split()].count()
data


# In[100]:


#pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(data)


# In[87]:


data=df2.groupby(["HighBloodPressure", "Stroke"])["Stroke".split()].count()
data


# In[88]:


data=df2.groupby(["HighBloodCholesterol", "Stroke"])["Stroke".split()].count()
data


# In[89]:


data=df2.groupby(["HeartAttack", "Stroke"])["Stroke".split()].count()
data


# In[90]:


data=df2.groupby(["MentalHealth", "Stroke"])["Stroke".split()].count()
data


# In[91]:


data=df2.groupby(["DepressiveDisorder", "Stroke"])["Stroke".split()].count()
data


# In[92]:


data=df2.groupby(["Overweight/Obesity", "Stroke"])["Stroke".split()].count()
data


# In[93]:


data=df2.groupby(["Smoke100", "Stroke"])["Stroke".split()].count()
data


# In[94]:


data=df2.groupby(["BingeDrinkers", "Stroke"])["Stroke".split()].count()
data


# In[95]:


data=df2.groupby(["PhysicalActivity", "Stroke"])["Stroke".split()].count()
data


# In[96]:


data=df2.groupby(["ConsumeFruitsEveryDay", "Stroke"])["Stroke".split()].count()
data


# In[97]:


data=df2.groupby(["ConsumeVegetablesEveryDay", "Stroke"])["Stroke".split()].count()
data

