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
dataset2 = pd.read_csv('DataSet.csv', na_values = missing_value_formats)


# In[3]:


dataset2.head()


# In[4]:


dataset2.shape


# ## Checking Values in Each Variable

# ### Primary Key:

# In[5]:


dataset2["SEQNO"]


# In[6]:


dataset2["SEQNO"].duplicated().value_counts()


# ### Health:

# In[7]:


dataset2["GENHLTH"].value_counts()


# In[8]:


dataset2["PHYSHLTH"].value_counts()


# In[9]:


dataset2["MENTHLTH"].value_counts()


# In[10]:


dataset2["MISTMNT"].value_counts()


# In[11]:


dataset2["_RFHLTH"].value_counts()


# In[12]:


dataset2["_RFHYPE5"].value_counts()


# In[13]:


dataset2["MEDCOST"].value_counts()


# In[14]:


dataset2["CHECKUP1"].value_counts()


# In[15]:


dataset2["BPHIGH4"].value_counts()


# In[16]:


dataset2["TOLDHI2"].value_counts()


# In[17]:


dataset2["CVDINFR4"].value_counts()


# In[18]:


dataset2["CVDCRHD4"].value_counts()


# In[19]:


dataset2["CVDSTRK3"].value_counts()


# In[20]:


dataset2["ASTHMA3"].value_counts()


# In[21]:


dataset2["CHCSCNCR"].value_counts()


# In[22]:


dataset2["CHCOCNCR"].value_counts()


# In[23]:


dataset2["CHCCOPD1"].value_counts()


# In[24]:


dataset2["HAVARTH3"].value_counts()


# In[25]:


dataset2["ADDEPEV2"].value_counts()


# In[26]:


dataset2["CHCKIDNY"].value_counts()


# In[27]:


dataset2["DIABETE3"].value_counts()


# In[28]:


dataset2["PREDIAB1"].value_counts()


# In[29]:


dataset2["_RFCHOL"].value_counts()


# In[30]:


dataset2["_MICHD"].value_counts()


# In[31]:


dataset2["_LTASTH1"].value_counts()


# In[32]:


dataset2["_CASTHM1"].value_counts()


# In[33]:


dataset2["_ASTHMS1"].value_counts()


# In[34]:


dataset2["_DRDXAR1"].value_counts()


# In[35]:


dataset2["_BMI5CAT"].value_counts()


# In[36]:


dataset2["DECIDE"].value_counts()


# In[37]:


dataset2["BLIND"].value_counts()


# ### Personal Information:

# In[38]:


dataset2["SEX"].value_counts()


# In[39]:


dataset2["SXORIENT"].value_counts()


# In[40]:


dataset2["AGE"].value_counts()


# In[41]:


dataset2["_IMPAGE"].value_counts()


# In[42]:


dataset2["_AGEG5YR"].value_counts()


# In[43]:


dataset2["_AGE65YR"].value_counts()


# In[44]:


dataset2["_AGE80"].value_counts()


# In[45]:


dataset2["_AGE_G"].value_counts()


# In[46]:


dataset2["MARITAL"].value_counts()


# In[47]:


dataset2["_EDUCAG"].value_counts()


# In[48]:


dataset2["RENTHOM1"].value_counts()


# In[49]:


dataset2["CPDEMO1"].value_counts()


# In[50]:


dataset2["EMPLOY1"].value_counts()


# In[51]:


dataset2["NUMADULT"].value_counts()


# In[52]:


dataset2["CHILDREN"].value_counts()


# In[53]:


dataset2["_CHLDCNT"].value_counts()


# In[54]:


dataset2["INCOME2"].value_counts()


# In[55]:


dataset2["_INCOMG"].value_counts()


# In[56]:


dataset2["INTERNET"].value_counts()


# ### Smoke and Alcohol:

# In[57]:


dataset2["SMOKE100"].value_counts()


# In[58]:


dataset2["SMOKDAY2"].value_counts()


# In[59]:


dataset2["LASTSMK2"].value_counts()


# In[60]:


dataset2["USENOW3"].value_counts()


# In[61]:


dataset2["_SMOKER3"].value_counts()


# In[62]:


dataset2["_RFSMOK3"].value_counts()


# In[63]:


dataset2["_DRNKWEK"].value_counts()


# In[64]:


dataset2["_RFBING5"].value_counts()


# In[65]:


dataset2["DRNK3GE5"].value_counts()


# ### Fruits and Vegetables:

# In[66]:


dataset2["_FRUTSUM"].value_counts()


# In[67]:


dataset2["_VEGESUM"].value_counts()


# In[68]:


dataset2["_FRTLT1"].value_counts()


# In[69]:


dataset2["_VEGLT1"].value_counts()


# ### Exercise and Physical Activity:

# In[70]:


dataset2["EXERANY2"].value_counts()


# In[71]:


dataset2["EXRACT11"].value_counts()


# In[72]:


dataset2["EXEROFT1"].value_counts()


# In[73]:


dataset2["EXERHMM1"].value_counts()


# In[74]:


dataset2["EXRACT21"].value_counts()


# In[75]:


dataset2["_TOTINDA"].value_counts()


# In[76]:


dataset2["PA1MIN_"].value_counts()


# In[77]:


dataset2["PA1VIGM_"].value_counts()


# In[78]:


dataset2["_PACAT1"].value_counts()


# ### Vaccine and Test:

# In[79]:


dataset2["FLUSHOT6"].value_counts()


# In[80]:


dataset2["PNEUVAC3"].value_counts()


# In[81]:


dataset2["HIVTST6"].value_counts()


# In[82]:


dataset2.shape


# In[ ]:




