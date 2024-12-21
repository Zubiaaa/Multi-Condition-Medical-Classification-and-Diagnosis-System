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
df1 = pd.read_csv('DataSetResearchQuestion1.csv', na_values = missing_value_formats)


# In[3]:


df1.head()


# In[4]:


df1.shape


# In[5]:


df1["CVDCRHD4"].value_counts()


# In[6]:


df1["CVDCRHD4"].isnull().sum()


# In[7]:


df1=df1[~df1['CVDCRHD4'].isnull()]


# In[8]:


df1["CVDCRHD4"].isnull().sum()


# In[9]:


df1 = df1.rename(columns={'CVDCRHD4': 'HeartDisease'})


# In[10]:


df1.head()


# In[11]:


df1.shape


# In[12]:


df1["SEQNO"].duplicated().value_counts()


# In[13]:


df1["SEQNO"].isnull().sum()


# In[14]:


df1 = df1.rename(columns={'SEQNO': 'RespondentID'})


# In[111]:


#df1["O_STATE"].value_counts()


# In[112]:


#df1["O_STATE"].isnull().sum()


# In[113]:


#df1['O_STATE'] = df1['O_STATE'].fillna(df1['O_STATE'].value_counts().index[0])
#df1["O_STATE"].value_counts()


# In[114]:


#df1["O_STATE"].isnull().sum()


# In[115]:


#df1 = df1.rename(columns={'O_STATE': 'State'})


# In[20]:


df1["MENTHLTH"].value_counts()


# In[21]:


df1["MENTHLTH"].isnull().sum()


# In[22]:


df1["MENTHLTH"].replace("Number of days", "Poor", inplace=True)
df1["MENTHLTH"].replace("None", "Good", inplace=True)
df1['MENTHLTH'] = df1['MENTHLTH'].fillna(df1['MENTHLTH'].value_counts().index[0])
df1["MENTHLTH"].value_counts()


# In[23]:


df1["MENTHLTH"].isnull().sum()


# In[24]:


df1 = df1.rename(columns={'MENTHLTH': 'MentalHealth'})


# In[25]:


df1["ADDEPEV2"].value_counts()


# In[26]:


df1["ADDEPEV2"].isnull().sum()


# In[27]:


df1['ADDEPEV2'] = df1['ADDEPEV2'].fillna(df1['ADDEPEV2'].value_counts().index[0])
df1["ADDEPEV2"].value_counts()


# In[28]:


df1["ADDEPEV2"].isnull().sum()


# In[29]:


df1 = df1.rename(columns={'ADDEPEV2': 'DepressiveDisorder'})


# In[30]:


df1["_RFHYPE5"].value_counts()


# In[31]:


df1["_RFHYPE5"].isnull().sum()


# In[32]:


df1['_RFHYPE5'] = df1['_RFHYPE5'].fillna(df1['_RFHYPE5'].value_counts().index[0])
df1["_RFHYPE5"].value_counts()


# In[33]:


df1["_RFHYPE5"].isnull().sum()


# In[34]:


df1 = df1.rename(columns={'_RFHYPE5': 'HighBloodPressure'})


# In[35]:


df1["TOLDHI2"].value_counts()


# In[36]:


df1["TOLDHI2"].isnull().sum()


# In[37]:


df1['TOLDHI2'] = df1['TOLDHI2'].fillna(df1['TOLDHI2'].value_counts().index[0])
df1["TOLDHI2"].value_counts()


# In[38]:


df1["TOLDHI2"].isnull().sum()


# In[39]:


df1 = df1.rename(columns={'TOLDHI2': 'HighBloodCholesterol'})


# In[40]:


df1["CHCCOPD1"].value_counts()


# In[41]:


df1["CHCCOPD1"].isnull().sum()


# In[42]:


df1['CHCCOPD1'] = df1['CHCCOPD1'].fillna(df1['CHCCOPD1'].value_counts().index[0])
df1["CHCCOPD1"].value_counts()


# In[43]:


df1["CHCCOPD1"].isnull().sum()


# In[44]:


df1 = df1.rename(columns={'CHCCOPD1': 'COPD'})


# In[45]:


df1["CHCKIDNY"].value_counts()


# In[46]:


df1["CHCKIDNY"].isnull().sum()


# In[47]:


df1['CHCKIDNY'] = df1['CHCKIDNY'].fillna(df1['CHCKIDNY'].value_counts().index[0])
df1["CHCKIDNY"].value_counts()


# In[48]:


df1["CHCKIDNY"].isnull().sum()


# In[49]:


df1 = df1.rename(columns={'CHCKIDNY': 'KidneyDisease'})


# In[50]:


df1["_RFBMI5"].value_counts()


# In[51]:


df1["_RFBMI5"].isnull().sum()


# In[52]:


df1['_RFBMI5'] = df1['_RFBMI5'].fillna(df1['_RFBMI5'].value_counts().index[0])
df1["_RFBMI5"].value_counts()


# In[53]:


df1["_RFBMI5"].isnull().sum()


# In[54]:


df1 = df1.rename(columns={'_RFBMI5': 'Overweight/Obesity'})


# In[55]:


df1["SEX"].value_counts()


# In[56]:


df1["SEX"].isnull().sum()


# In[57]:


df1 = df1.rename(columns={'SEX': 'Sex'})


# In[58]:


df1["AGE"].value_counts()


# In[59]:


df1["AGE"].isnull().sum()


# In[60]:


df1["AGE"].replace("Age 65 or older", "65 or older", inplace=True)
df1["AGE"].replace("Age 55 - 64", "55 - 64", inplace=True)
df1["AGE"].replace("Age 45 - 54", "45 - 54", inplace=True)
df1["AGE"].replace("Age 35 - 44", "35 - 44", inplace=True)
df1["AGE"].replace("Age 25 - 34", "25 - 34", inplace=True)
df1["AGE"].replace("Age 18 - 24", "18 - 24", inplace=True)
df1['AGE'] = df1['AGE'].fillna(df1['AGE'].value_counts().index[0])
df1["AGE"].value_counts()


# In[61]:


df1["AGE"].isnull().sum()


# In[62]:


df1 = df1.rename(columns={'AGE': 'AgeRange'})


# In[63]:


df1["SMOKE100"].value_counts()


# In[64]:


df1["SMOKE100"].isnull().sum()


# In[65]:


df1['SMOKE100'] = df1['SMOKE100'].fillna(df1['SMOKE100'].value_counts().index[0])
df1["SMOKE100"].value_counts()


# In[66]:


df1["SMOKE100"].isnull().sum()


# In[67]:


df1 = df1.rename(columns={'SMOKE100': 'Smoke100'})


# In[68]:


df1["_RFBING5"].value_counts()


# In[69]:


df1["_RFBING5"].isnull().sum()


# In[70]:


df1['_RFBING5'] = df1['_RFBING5'].fillna(df1['_RFBING5'].value_counts().index[0])
df1["_RFBING5"].value_counts()


# In[71]:


df1["_RFBING5"].isnull().sum()


# In[72]:


df1 = df1.rename(columns={'_RFBING5': 'BingeDrinkers'})


# In[73]:


df1["EXERANY2"].value_counts()


# In[74]:


df1["EXERANY2"].isnull().sum()


# In[75]:


df1['EXERANY2'] = df1['EXERANY2'].fillna(df1['EXERANY2'].value_counts().index[0])
df1["EXERANY2"].value_counts()


# In[76]:


df1["EXERANY2"].isnull().sum()


# In[77]:


df1 = df1.rename(columns={'EXERANY2': 'PhysicalActivity'})


# In[78]:


df1["_FRTLT1"].value_counts()


# In[79]:


df1["_FRTLT1"].isnull().sum()


# In[80]:


df1["_FRTLT1"].replace("Consumed fruit one or more times per day", "Yes", inplace=True)
df1["_FRTLT1"].replace("Consumed fruit less than one time per day", "No", inplace=True)
df1['_FRTLT1'] = df1['_FRTLT1'].fillna(df1['_FRTLT1'].value_counts().index[0])
df1["_FRTLT1"].value_counts()


# In[81]:


df1["_FRTLT1"].isnull().sum()


# In[82]:


df1 = df1.rename(columns={'_FRTLT1': 'ConsumeFruitsEveryDay'})


# In[83]:


df1["_VEGLT1"].value_counts()


# In[84]:


df1["_VEGLT1"].isnull().sum()


# In[85]:


df1["_VEGLT1"].replace("Consumed vegetables one or more times per day", "Yes", inplace=True)
df1["_VEGLT1"].replace("Consumed vegetables less than one time per day", "No", inplace=True)
df1['_VEGLT1'] = df1['_VEGLT1'].fillna(df1['_VEGLT1'].value_counts().index[0])
df1["_VEGLT1"].value_counts()


# In[86]:


df1["_VEGLT1"].isnull().sum()


# In[87]:


df1 = df1.rename(columns={'_VEGLT1': 'ConsumeVegetablesEveryDay'})


# In[88]:


df1.head()


# In[89]:


df1.shape


# In[90]:


# Exporting dataset:
from IPython.display import FileLink, FileLinks

df1.to_csv('ResearchQuestion1.csv', index=False)


# ## Relationship:

# In[109]:


data=df1.groupby(["AgeRange", "Sex", "HeartDisease"])["HeartDisease".split()].count()
data


# In[110]:


#pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(data)


# In[106]:


data=df1.groupby(["AgeRange", "Sex", "HighBloodPressure", "HeartDisease"])["HeartDisease".split()].count()
data


# In[116]:


data=df1.groupby(["AgeRange", "Sex", "HighBloodCholesterol", "HeartDisease"])["HeartDisease".split()].count()
data


# In[117]:


data=df1.groupby(["AgeRange", "Sex", "MentalHealth", "HeartDisease"])["HeartDisease".split()].count()
data


# In[118]:


data=df1.groupby(["AgeRange", "Sex", "DepressiveDisorder", "HeartDisease"])["HeartDisease".split()].count()
data


# In[119]:


data=df1.groupby(["AgeRange", "Sex", "COPD", "HeartDisease"])["HeartDisease".split()].count()
data


# In[120]:


data=df1.groupby(["AgeRange", "Sex", "KidneyDisease", "HeartDisease"])["HeartDisease".split()].count()
data


# In[121]:


data=df1.groupby(["AgeRange", "Sex", "Overweight/Obesity", "HeartDisease"])["HeartDisease".split()].count()
data


# In[122]:


data=df1.groupby(["AgeRange", "Sex", "Smoke100", "HeartDisease"])["HeartDisease".split()].count()
data


# In[123]:


data=df1.groupby(["AgeRange", "Sex", "BingeDrinkers", "HeartDisease"])["HeartDisease".split()].count()
data


# In[124]:


data=df1.groupby(["AgeRange", "Sex", "PhysicalActivity", "HeartDisease"])["HeartDisease".split()].count()
data


# In[125]:


data=df1.groupby(["AgeRange", "Sex", "ConsumeFruitsEveryDay", "HeartDisease"])["HeartDisease".split()].count()
data


# In[126]:


data=df1.groupby(["AgeRange", "Sex", "ConsumeVegetablesEveryDay", "HeartDisease"])["HeartDisease".split()].count()
data


# In[ ]:




