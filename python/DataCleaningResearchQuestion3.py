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
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr


# In[2]:


missing_value_formats = ["Don't know/Not Sure/Refused/Missing", "Don´t know, refused or missing values", "Don't know/Not sure/Missing", "Don´t know/Not Sure/Refused/Missing", "Don't know/Not Sure Or Refused/Missing", "Don't know/Refused/Missing", "Missing", "Don´t Know/Not Sure", "Don't Know / Not Sure", "Don't Know/Not sure", "Don´t Know / Not sure", "Refused","Don't know/Not sure","Not asked or Missing","Don't know/Not Sure","Don't know / Not sure","Don't know / Not Sure", "Data do not meet the criteria for statistical reliability, data quality, or confidentiality (data are suppressed).", "D"]
df3 = pd.read_csv('DataSetResearchQuestion3.csv', na_values = missing_value_formats)


# In[3]:


df3.head()


# In[4]:


df3.shape


# In[5]:


df3["MENTHLTH"].value_counts()


# In[6]:


df3["MENTHLTH"].isnull().sum()


# In[7]:


df3["MENTHLTH"].replace("Number of days", "Poor", inplace=True)
df3["MENTHLTH"].replace("None", "Good", inplace=True)
df3=df3[~df3['MENTHLTH'].isnull()]
df3["MENTHLTH"].isnull().sum()


# In[8]:


df3["MENTHLTH"].value_counts()


# In[9]:


df3 = df3.rename(columns={'MENTHLTH': 'MentalHealth'})


# In[10]:


df3["SEQNO"].isnull().sum()


# In[11]:


df3["SEQNO"].duplicated().value_counts()


# In[12]:


df3 = df3.rename(columns={'SEQNO': 'RespondentID'})


# In[13]:


df3["O_STATE"].value_counts()


# In[14]:


df3["O_STATE"].isnull().sum()


# In[15]:


df3['O_STATE'] = df3['O_STATE'].fillna(df3['O_STATE'].value_counts().index[0])
df3["O_STATE"].value_counts()


# In[16]:


df3["O_STATE"].isnull().sum()


# In[17]:


df3 = df3.rename(columns={'O_STATE': 'State'})


# In[18]:


df3["SEX"].value_counts()


# In[19]:


df3["SEX"].isnull().sum()


# In[20]:


df3 = df3.rename(columns={'SEX': 'Sex'})


# In[21]:


df3["INCOME2"].value_counts()


# In[22]:


df3["INCOME2"].isnull().sum()


# In[23]:


df3["INCOME2"].replace("Less than $75,000 ($50,000 to less than $75,000)", "$50,000 to less than $75,000", inplace=True)
df3["INCOME2"].replace("Less than $50,000 ($35,000 to less than $50,000)", "$35,000 to less than $50,000", inplace=True)
df3["INCOME2"].replace("Less than $35,000 ($25,000 to less than $35,000)", "$25,000 to less than $35,000", inplace=True)
df3["INCOME2"].replace("Less than $25,000 ($20,000 to less than $25,000)", "$20,000 to less than $25,000", inplace=True)
df3["INCOME2"].replace("Less than $20,000 ($15,000 to less than $20,000)", "$15,000 to less than $20,000", inplace=True)
df3["INCOME2"].replace("Less than $15,000 ($10,000 to less than $15,000)", "$10,000 to less than $15,000", inplace=True)
df3['INCOME2'] = df3['INCOME2'].fillna(df3['INCOME2'].value_counts().index[0])
df3["INCOME2"].value_counts()


# In[24]:


df3["INCOME2"].isnull().sum()


# In[25]:


df3 = df3.rename(columns={'INCOME2': 'IncomeRange'})


# In[26]:


df3["PHYSHLTH"].value_counts()


# In[27]:


df3["PHYSHLTH"].isnull().sum()


# In[28]:


df3["PHYSHLTH"].replace("Number of days", "Poor", inplace=True)
df3["PHYSHLTH"].replace("None", "Good", inplace=True)
df3['PHYSHLTH'] = df3['PHYSHLTH'].fillna(df3['PHYSHLTH'].value_counts().index[0])
df3["PHYSHLTH"].value_counts()


# In[29]:


df3["PHYSHLTH"].isnull().sum()


# In[30]:


df3 = df3.rename(columns={'PHYSHLTH': 'PhysicalHealth'})


# In[31]:


df3.shape


# In[32]:


df3["ADDEPEV2"].value_counts()


# In[33]:


df3["ADDEPEV2"].isnull().sum()


# In[34]:


df3['ADDEPEV2'] = df3['ADDEPEV2'].fillna(df3['ADDEPEV2'].value_counts().index[0])
df3["ADDEPEV2"].value_counts()


# In[35]:


df3["ADDEPEV2"].isnull().sum()


# In[36]:


df3 = df3.rename(columns={'ADDEPEV2': 'DepressiveDisorder'})


# In[37]:


df3["MEDCOST"].value_counts()


# In[38]:


df3["MEDCOST"].isnull().sum()


# In[39]:


df3['MEDCOST'] = df3['MEDCOST'].fillna(df3['MEDCOST'].value_counts().index[0])
df3["MEDCOST"].value_counts()


# In[40]:


df3["MEDCOST"].isnull().sum()


# In[41]:


df3 = df3.rename(columns={'MEDCOST': 'UnaffordMedicalCost'})


# In[42]:


df3["MEDSCOST"].value_counts()


# In[43]:


df3["MEDSCOST"].isnull().sum()


# In[44]:


df3["MEDSCOST"].replace("No medication was prescribed", np.nan, inplace=True)
df3['MEDSCOST'] = df3['MEDSCOST'].fillna(df3['MEDSCOST'].value_counts().index[0])
df3["MEDSCOST"].value_counts()


# In[45]:


df3["MEDSCOST"].isnull().sum()


# In[46]:


df3 = df3.rename(columns={'MEDSCOST': 'UnaffordPrescribedMedication'})


# In[47]:


df3["SMOKE100"].value_counts()


# In[48]:


df3["SMOKE100"].isnull().sum()


# In[49]:


df3['SMOKE100'] = df3['SMOKE100'].fillna(df3['SMOKE100'].value_counts().index[0])
df3["SMOKE100"].value_counts()


# In[50]:


df3["SMOKE100"].isnull().sum()


# In[51]:


df3 = df3.rename(columns={'SMOKE100': 'Smoke100'})


# In[52]:


df3["_RFBING5"].value_counts()


# In[53]:


df3["_RFBING5"].isnull().sum()


# In[54]:


df3['_RFBING5'] = df3['_RFBING5'].fillna(df3['_RFBING5'].value_counts().index[0])
df3["_RFBING5"].value_counts()


# In[55]:


df3["_RFBING5"].isnull().sum()


# In[56]:


df3 = df3.rename(columns={'_RFBING5': 'BingeDrinkers'})


# In[57]:


df3.head()


# In[58]:


df3.shape


# In[59]:


# Exporting dataset:
from IPython.display import FileLink, FileLinks

df3.to_csv('ResearchQuestion3.csv', index=False)


# ## Relationship:

# In[70]:


data=df3.groupby(["Sex", "State", "MentalHealth"])["MentalHealth".split()].count()


# In[71]:


pd.set_option("display.max_rows", None, "display.max_columns", None)
print(data)


# In[72]:


data=df3.groupby(["Sex", "State", "DepressiveDisorder", "MentalHealth"])["MentalHealth".split()].count()
data


# In[73]:


data=df3.groupby(["Sex", "State", "Smoke100", "MentalHealth"])["MentalHealth".split()].count()
data


# In[74]:


data=df3.groupby(["Sex", "State", "BingeDrinkers", "MentalHealth"])["MentalHealth".split()].count()
data


# In[75]:


data=df3.groupby(["Sex", "State", "IncomeRange", "MentalHealth"])["MentalHealth".split()].count()
data


# In[76]:


data=df3.groupby(["Sex", "State", "PhysicalHealth", "MentalHealth"])["MentalHealth".split()].count()
data


# In[77]:


data=df3.groupby(["Sex", "State", "UnaffordMedicalCost", "MentalHealth"])["MentalHealth".split()].count()
data


# In[78]:


data=df3.groupby(["Sex", "State", "UnaffordPrescribedMedication", "MentalHealth"])["MentalHealth".split()].count()
data


# In[ ]:




