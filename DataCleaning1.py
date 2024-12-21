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


missing_value_formats = ["Don't know/Not Sure Or Refused/Missing", "Don't know/Refused/Missing", "Missing", "Don´t Know/Not Sure", "Don't Know / Not Sure", "Don't Know/Not sure", "Don´t Know / Not sure", "Refused","Don't know/Not sure","Not asked or Missing","Don't know/Not Sure","Don't know / Not sure","Don't know / Not Sure", "Data do not meet the criteria for statistical reliability, data quality, or confidentiality (data are suppressed).", "D"]
dataset = pd.read_csv('brfss_for_bda_2021.csv', na_values = missing_value_formats)


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.dtypes


# In[6]:


dataset["_STATE"].value_counts()


# In[7]:


dataset["_STATE"].isnull().sum()


# In[8]:


dataset["_GEOSTR"].value_counts()


# In[9]:


dataset["_GEOSTR"].isnull().sum()


# In[10]:


dataset["_DENSTR2"].value_counts()


# In[11]:


dataset.drop("_DENSTR2", axis=1, inplace=True)


# In[12]:


dataset["PRECALL"].value_counts()


# In[13]:


dataset['PRECALL'].replace(to_replace ="cell phone - unknown status",
                 value ="To be called", inplace=True)


# In[14]:


dataset["PRECALL"].value_counts()


# In[15]:


dataset["PRECALL"].isnull().sum()


# In[16]:


dataset["SECSCRFL"].value_counts()


# In[17]:


dataset["SECSCRFL"].isnull().sum()


# In[18]:


dataset["REPNUM"].value_counts()


# In[19]:


dataset["REPNUM"].isnull().sum()


# In[20]:


dataset["REPDEPTH"].value_counts()


# In[21]:


dataset["REPDEPTH"].isnull().sum()


# In[22]:


dataset["FMONTH"].value_counts()


# In[23]:


dataset["FMONTH"].isnull().sum()


# In[24]:


dataset["IDATE"].value_counts()


# In[25]:


dataset["IMONTH"].value_counts()


# In[26]:


dataset["IDAY"].value_counts()


# In[27]:


dataset["IYEAR"].value_counts()


# In[28]:


dataset["DISPCODE"].value_counts()


# In[29]:


dataset["SEQNO"].value_counts()


# In[30]:


dataset["_PSU"].value_counts()


# In[31]:


dataset["NATTMPTS"].value_counts()


# In[32]:


dataset["NRECSEL"].value_counts()


# In[33]:


dataset["NRECSTR"].value_counts()


# In[34]:


dataset["PVTRESD1"].value_counts()


# In[35]:


dataset["PVTRESD1"].replace("Missing", np.nan, inplace=True)


# In[36]:


dataset["PVTRESD1"].value_counts()


# In[37]:


dataset["PVTRESD1"].isnull().sum()


# In[38]:


dataset["COLGHOUS"].value_counts()


# In[39]:


dataset.drop("COLGHOUS", axis=1, inplace=True)


# In[40]:


dataset["STATERES"].value_counts()


# In[41]:


dataset["STATERES"].isnull().sum()


# In[42]:


dataset["CELLFON3"].value_counts()


# In[43]:


dataset["CELLFON3"].replace("Missing", np.nan, inplace=True)


# In[44]:


dataset["CELLFON3"].value_counts()


# In[45]:


dataset["CELLFON3"].isnull().sum()


# In[46]:


dataset["LADULT"].value_counts()


# In[47]:


dataset.drop("LADULT", axis=1, inplace=True)


# In[48]:


dataset["NUMADULT"].value_counts()


# In[49]:


dataset["NUMADULT"].isnull().sum()


# In[50]:


dataset["CADULT"].value_counts()


# In[51]:


dataset.drop("CADULT", axis=1, inplace=True)


# In[52]:


dataset["CCLGHOUS"].value_counts()


# In[53]:


dataset.drop("CCLGHOUS", axis=1, inplace=True)


# In[54]:


dataset["CSTATE"].value_counts()


# In[55]:


dataset.drop("CSTATE", axis=1, inplace=True)


# In[56]:


dataset["RSPSTATE"].value_counts()


# In[57]:


dataset.drop("RSPSTATE", axis=1, inplace=True)


# In[58]:


dataset["LANDLINE"].value_counts()


# In[59]:


dataset.drop("LANDLINE", axis=1, inplace=True)


# In[60]:


dataset["HHADULT"].value_counts()


# In[61]:


dataset["HHADULT"].isnull().sum()


# In[62]:


dataset.drop("HHADULT", axis=1, inplace=True)


# In[63]:


dataset["GENHLTH"].value_counts()


# In[64]:


dataset["GENHLTH"].replace("Don't know/Not Sure", "Very good", inplace=True)


# In[65]:


dataset["GENHLTH"].value_counts()


# In[66]:


dataset["GENHLTH"].value_counts()


# In[67]:


dataset["GENHLTH"].isnull().sum()


# In[68]:


dataset["PHYSHLTH"].value_counts()


# In[69]:


dataset["PHYSHLTH"].isnull().sum()


# In[70]:


dataset["MENTHLTH"].value_counts()


# In[71]:


dataset["GENHLTH"].replace("Refused", np.nan, inplace=True)
dataset["MENTHLTH"].replace("Refused", np.nan, inplace=True)
dataset["MENTHLTH"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[72]:


dataset["MENTHLTH"].value_counts()


# In[73]:


dataset["MENTHLTH"].isnull().sum()


# In[74]:


dataset["GENHLTH"].value_counts()


# In[75]:


dataset["POORHLTH"].value_counts()


# In[76]:


dataset["POORHLTH"].replace("Not asked or Missing", np.nan, inplace=True)
dataset["POORHLTH"].replace("Refused", np.nan, inplace=True)
dataset["POORHLTH"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[77]:


dataset["POORHLTH"].value_counts()


# In[78]:


dataset["POORHLTH"].isnull().sum()


# In[79]:


dataset["HLTHPLN1"].value_counts()


# In[80]:


dataset["HLTHPLN1"].replace("Refused", np.nan, inplace=True)
dataset["HLTHPLN1"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[81]:


dataset["HLTHPLN1"].value_counts()


# In[82]:


dataset["HLTHPLN1"].isnull().sum()


# In[83]:


dataset["PERSDOC2"].value_counts()


# In[84]:


dataset["PERSDOC2"].replace("Refused", np.nan, inplace=True)
dataset["PERSDOC2"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[85]:


dataset["PERSDOC2"].value_counts()


# In[86]:


dataset["MEDCOST"].value_counts()


# In[87]:


dataset["MEDCOST"].replace("Refused", np.nan, inplace=True)
dataset["MEDCOST"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[88]:


dataset["MEDCOST"].value_counts()


# In[89]:


dataset["CHECKUP1"].value_counts()


# In[90]:


dataset["CHECKUP1"].replace("Refused", np.nan, inplace=True)
dataset["CHECKUP1"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[91]:


dataset["CHECKUP1"].value_counts()


# In[92]:


dataset["BPHIGH4"].value_counts()


# In[93]:


dataset["BPHIGH4"].replace("Refused", np.nan, inplace=True)
dataset["BPHIGH4"].replace("Don´t know/Not Sure", np.nan, inplace=True)


# In[94]:


dataset["BPHIGH4"].value_counts()


# In[977]:


dataset["BPHIGH4"].isnull().sum()


# In[95]:


dataset["BPMEDS"].value_counts()


# In[96]:


dataset["BPMEDS"].replace("Refused", np.nan, inplace=True)
dataset["BPMEDS"].replace("Don't know/Not Sure", np.nan, inplace=True)
dataset["BPMEDS"].replace("Not asked or Missing", np.nan, inplace=True)


# In[97]:


dataset["BPMEDS"].value_counts()


# In[98]:


dataset["BPMEDS"].isnull().sum()


# In[99]:


dataset.drop("BPMEDS", axis=1, inplace=True)


# In[100]:


dataset["BLOODCHO"].value_counts()


# In[101]:


dataset["BLOODCHO"].replace("Refused", np.nan, inplace=True)
dataset["BLOODCHO"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[102]:


dataset["BLOODCHO"].value_counts()


# In[103]:


dataset["CHOLCHK"].value_counts()


# In[104]:


dataset["CHOLCHK"].replace("Refused", np.nan, inplace=True)
dataset["CHOLCHK"].replace("Don't know/Not Sure", np.nan, inplace=True)
dataset["CHOLCHK"].replace("Not asked or Missing", np.nan, inplace=True)


# In[105]:


dataset["CHOLCHK"].value_counts()


# In[106]:


dataset["TOLDHI2"].value_counts()


# In[107]:


dataset["TOLDHI2"].replace("Not asked or Missing", np.nan, inplace=True)
dataset["TOLDHI2"].replace("Refused", np.nan, inplace=True)
dataset["TOLDHI2"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[108]:


dataset["TOLDHI2"].value_counts()


# In[109]:


dataset["TOLDHI2"].isnull().sum()


# In[110]:


dataset["CVDINFR4"].value_counts()


# In[111]:


dataset["CVDINFR4"].replace("Refused", np.nan, inplace=True)
dataset["CVDINFR4"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[112]:


dataset["CVDINFR4"].value_counts()


# In[113]:


dataset["CVDINFR4"].isnull().sum()


# In[114]:


dataset["CVDCRHD4"].value_counts()


# In[115]:


dataset["CVDCRHD4"].replace("Refused", np.nan, inplace=True)
dataset["CVDCRHD4"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[116]:


dataset["CVDCRHD4"].value_counts()


# In[117]:


dataset["CVDCRHD4"].isnull().sum()


# In[118]:


dataset["CVDSTRK3"].value_counts()


# In[119]:


dataset["CVDSTRK3"].replace("Refused", np.nan, inplace=True)
dataset["CVDSTRK3"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[120]:


dataset["CVDSTRK3"].value_counts()


# In[121]:


dataset["CVDSTRK3"].isnull().sum()


# In[122]:


dataset["ASTHMA3"].value_counts()


# In[123]:


dataset["ASTHMA3"].replace("Refused", np.nan, inplace=True)
dataset["ASTHMA3"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[124]:


dataset["ASTHMA3"].value_counts()


# In[125]:


dataset["ASTHMA3"].isnull().sum()


# In[126]:


dataset["ASTHNOW"].value_counts()


# In[127]:


dataset["ASTHNOW"].replace("Not asked or Missing", np.nan, inplace=True)
dataset["ASTHNOW"].replace("Refused", np.nan, inplace=True)
dataset["ASTHNOW"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[128]:


dataset["ASTHNOW"].value_counts()


# In[129]:


dataset["ASTHNOW"].isnull().sum()


# In[130]:


dataset.drop("ASTHNOW", axis=1, inplace=True)


# In[131]:


dataset["CHCSCNCR"].value_counts()


# In[132]:


dataset["CHCSCNCR"].replace("Refused", np.nan, inplace=True)
dataset["CHCSCNCR"].replace("Don't know / Not sure", np.nan, inplace=True)


# In[133]:


dataset["CHCSCNCR"].value_counts()


# In[134]:


dataset["CHCSCNCR"].isnull().sum()


# In[135]:


dataset["CHCCOPD1"].value_counts()


# In[136]:


dataset["CHCCOPD1"].replace("Refused", np.nan, inplace=True)
dataset["CHCCOPD1"].replace("Don't know / Not sure", np.nan, inplace=True)


# In[137]:


dataset["CHCCOPD1"].value_counts()


# In[138]:


dataset["CHCCOPD1"].isnull().sum()


# In[139]:


dataset["HAVARTH3"].value_counts()


# In[140]:


dataset["HAVARTH3"].replace("Refused", np.nan, inplace=True)
dataset["HAVARTH3"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[141]:


dataset["HAVARTH3"].value_counts()


# In[142]:


dataset["HAVARTH3"].isnull().sum()


# In[143]:


dataset["ADDEPEV2"].value_counts()


# In[144]:


dataset["ADDEPEV2"].replace("Refused", np.nan, inplace=True)
dataset["ADDEPEV2"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[145]:


dataset["ADDEPEV2"].value_counts()


# In[146]:


dataset["ADDEPEV2"].isnull().sum()


# In[147]:


dataset["CHCKIDNY"].value_counts()


# In[148]:


dataset["CHCKIDNY"].replace("Refused", np.nan, inplace=True)
dataset["CHCKIDNY"].replace("Don't know / Not sure", np.nan, inplace=True)


# In[149]:


dataset["CHCKIDNY"].value_counts()


# In[150]:


dataset["CHCKIDNY"].isnull().sum()


# In[151]:


dataset["DIABETE3"].value_counts()


# In[152]:


dataset["DIABETE3"].replace("Refused", np.nan, inplace=True)
dataset["DIABETE3"].replace("Don't know/Not Sure", np.nan, inplace=True)


# In[153]:


dataset["DIABETE3"].value_counts()


# In[154]:


dataset["DIABAGE2"].value_counts()


# In[155]:


dataset["DIABAGE2"].replace("Refused", np.nan, inplace=True)
dataset["DIABAGE2"].replace("Don't know/Not sure", np.nan, inplace=True)
dataset["DIABAGE2"].replace("Not asked or Missing", np.nan, inplace=True)


# In[156]:


dataset["DIABAGE2"].value_counts()


# In[157]:


dataset["SEX"].value_counts()


# In[158]:


dataset["SEX"].isnull().sum()


# In[159]:


dataset["AGE"].value_counts()


# In[160]:


dataset["AGE"].replace("Refused", np.nan, inplace=True)
dataset["AGE"].replace("Don't know/Not sure", np.nan, inplace=True)


# In[161]:


dataset["AGE"].value_counts()


# In[162]:


dataset["HISPANC3"].value_counts()


# In[163]:


dataset["HISPANC3"].isnull().sum()


# In[164]:


dataset.drop("HISPANC3", axis=1, inplace=True)


# In[165]:


dataset["MRACE1"].value_counts()


# In[166]:


dataset["MRACE1"].isnull().sum()


# In[167]:


dataset.drop("MRACE1", axis=1, inplace=True)


# In[168]:


dataset["ORACE3"].value_counts()


# In[169]:


dataset.drop("ORACE3", axis=1, inplace=True)


# In[170]:


dataset["MARITAL"].value_counts()


# In[171]:


dataset["MARITAL"].isnull().sum()


# In[172]:


dataset["EDUCA"].value_counts()


# In[173]:


dataset["EDUCA"].isnull().sum()


# In[174]:


dataset["RENTHOM1"].value_counts()


# In[978]:


dataset["RENTHOM1"].isnull().sum()


# In[175]:


dataset["CTYCODE1"].value_counts()


# In[176]:


dataset.drop("CTYCODE1", axis=1, inplace=True)


# In[ ]:





# In[ ]:





# In[177]:


dataset["ZIPCODE"].value_counts()


# In[178]:


dataset.drop("ZIPCODE", axis=1, inplace=True)


# In[179]:


dataset["NUMHHOL2"].value_counts()


# In[180]:


dataset["NUMHHOL2"].isnull().sum()


# In[181]:


dataset["NUMPHON2"].value_counts()


# In[182]:


dataset["NUMPHON2"].isnull().sum()


# In[183]:


dataset.drop("NUMPHON2", axis=1, inplace=True)


# In[184]:


dataset["CPDEMO1"].value_counts()


# In[185]:


dataset["CPDEMO1"].isnull().sum()


# In[186]:


dataset["VETERAN3"].value_counts()


# In[187]:


dataset["VETERAN3"].isnull().sum()


# In[188]:


dataset["EMPLOY1"].value_counts()


# In[189]:


dataset["CHILDREN"].value_counts()


# In[190]:


dataset["CHILDREN"].isnull().sum()


# In[191]:


dataset["INCOME2"].value_counts()


# In[192]:


dataset["INTERNET"].value_counts()


# In[193]:


dataset["INTERNET"].isnull().sum()


# In[194]:


dataset["WEIGHT2"].value_counts()


# In[195]:


dataset["WEIGHT2"].isnull().sum()


# In[196]:


dataset.drop("WEIGHT2", axis=1, inplace=True)


# In[ ]:





# In[ ]:





# In[197]:


dataset["HEIGHT3"].value_counts()


# In[198]:


dataset["HEIGHT3"].isnull().sum()


# In[199]:


dataset.drop("HEIGHT3", axis=1, inplace=True)


# In[200]:


dataset["PREGNANT"].value_counts()


# In[201]:


dataset["PREGNANT"].isnull().sum()


# In[202]:


dataset.drop("PREGNANT" ,axis=1 ,inplace=True)


# In[203]:


dataset["QLACTLM2"].value_counts()


# In[204]:


dataset["QLACTLM2"].isnull().sum()


# In[205]:


dataset["USEEQUIP"].value_counts()


# In[206]:


dataset["USEEQUIP"].isnull().sum()


# In[207]:


dataset["BLIND"].value_counts()


# In[208]:


dataset["BLIND"].isnull().sum()


# In[209]:


dataset["DECIDE"].value_counts()


# In[210]:


dataset["DECIDE"].isnull().sum()


# In[211]:


dataset["DIFFWALK"].value_counts()


# In[212]:


dataset["DIFFWALK"].isnull().sum()


# In[213]:


dataset["DIFFDRES"].value_counts()


# In[214]:


dataset["DIFFDRES"].isnull().sum()


# In[215]:


dataset["DIFFALON"].value_counts()


# In[216]:


dataset["DIFFALON"].isnull().sum()


# In[217]:


dataset["SMOKE100"].value_counts()


# In[218]:


dataset["SMOKE100"].isnull().sum()


# In[219]:


dataset["SMOKDAY2"].value_counts()


# In[220]:


dataset["STOPSMK2"].value_counts()


# In[221]:


dataset["STOPSMK2"].isnull().sum()


# In[222]:


dataset.drop("STOPSMK2", axis=1, inplace=True)


# In[223]:


dataset["LASTSMK2"].value_counts()


# In[224]:


dataset["USENOW3"].value_counts()


# In[225]:


dataset["ALCDAY5"].value_counts()


# In[226]:


dataset["AVEDRNK2"].value_counts()


# In[227]:


dataset["AVEDRNK2"].isnull().sum()


# In[228]:


dataset["DRNK3GE5"].value_counts()


# In[229]:


dataset["DRNK3GE5"].isnull().sum()


# In[230]:


dataset["MAXDRNKS"].value_counts()


# In[231]:


dataset["MAXDRNKS"].isnull().sum()


# In[232]:


dataset["FRUITJU1"].value_counts()


# In[233]:


dataset["FRUIT1"].value_counts()


# In[234]:


dataset["FVBEANS"].value_counts()


# In[235]:


dataset["FVGREEN"].value_counts()


# In[236]:


dataset["FVORANG"].value_counts()


# In[237]:


dataset["VEGETAB1"].value_counts()


# In[238]:


dataset["EXERANY2"].value_counts()


# In[239]:


dataset["EXERANY2"].isnull().sum()


# In[240]:


dataset["EXRACT11"].value_counts()


# In[241]:


dataset["EXEROFT1"].value_counts()


# In[242]:


dataset["EXEROFT1"].isnull().sum()


# In[243]:


dataset["EXERHMM1"].value_counts()


# In[244]:


dataset["EXERHMM1"].isnull().sum()


# In[245]:


dataset["EXRACT21"].value_counts()


# In[246]:


dataset["EXEROFT2"].value_counts()


# In[247]:


dataset["EXEROFT2"].isnull().sum()


# In[248]:


dataset.drop("EXEROFT2", axis=1, inplace=True)


# In[249]:


dataset["EXERHMM2"].value_counts()


# In[250]:


dataset["EXERHMM2"].isnull().sum()


# In[251]:


dataset.drop("EXERHMM2", axis=1, inplace=True)


# In[252]:


dataset["STRENGTH"].value_counts()


# In[253]:


dataset["LMTJOIN3"].value_counts()


# In[254]:


dataset["LMTJOIN3"].isnull().sum()


# In[255]:


dataset.drop("LMTJOIN3", axis=1, inplace=True)


# In[256]:


dataset["ARTHDIS2"].value_counts()


# In[257]:


dataset["ARTHDIS2"].isnull().sum()


# In[258]:


dataset.drop("ARTHDIS2", axis=1, inplace=True)


# In[259]:


dataset["ARTHSOCL"].value_counts()


# In[260]:


dataset["ARTHSOCL"].isnull().sum()


# In[261]:


dataset.drop("ARTHSOCL", axis=1, inplace=True)


# In[262]:


dataset["JOINPAIN"].value_counts()


# In[263]:


dataset["JOINPAIN"].isnull().sum()


# In[264]:


dataset.drop("JOINPAIN", axis=1, inplace=True)


# In[265]:


dataset["SEATBELT"].value_counts()


# In[266]:


dataset["SEATBELT"].isnull().sum()


# In[267]:


dataset["FLUSHOT6"].value_counts()


# In[268]:


dataset["FLUSHOT6"].isnull().sum()


# In[269]:


dataset["FLSHTMY2"].value_counts()


# In[270]:


dataset["FLSHTMY2"].isnull().sum()


# In[271]:


dataset.drop("FLSHTMY2", axis=1, inplace=True)


# In[272]:


dataset["IMFVPLAC"].value_counts()


# In[273]:


dataset["IMFVPLAC"].isnull().sum()


# In[274]:


dataset.drop("IMFVPLAC", axis=1, inplace=True)


# In[275]:


dataset["PNEUVAC3"].value_counts()


# In[276]:


dataset["PNEUVAC3"].isnull().sum()


# In[277]:


dataset["HIVTST6"].value_counts()


# In[278]:


dataset["HIVTST6"].isnull().sum()


# In[279]:


dataset["HIVTSTD3"].value_counts()


# In[280]:


dataset["HIVTSTD3"].isnull().sum()


# In[281]:


dataset.drop("HIVTSTD3", axis=1, inplace=True)


# In[282]:


dataset["WHRTST10"].value_counts()


# In[283]:


dataset["WHRTST10"].isnull().sum()


# In[284]:


dataset.drop("WHRTST10", axis=1, inplace=True)


# In[285]:


dataset["PDIABTST"].value_counts()


# In[286]:


dataset["PDIABTST"].isnull().sum()


# In[287]:


dataset.drop("PDIABTST", axis=1, inplace=True)


# In[288]:


dataset["PREDIAB1"].value_counts()


# In[289]:


dataset["PREDIAB1"].isnull().sum()


# In[290]:


dataset.drop("PREDIAB1", axis=1, inplace=True)


# In[291]:


dataset["INSULIN"].value_counts()


# In[292]:


dataset["INSULIN"].isnull().sum()


# In[293]:


dataset.drop("INSULIN", axis=1, inplace=True)


# In[294]:


dataset["BLDSUGAR"].value_counts()


# In[295]:


dataset["BLDSUGAR"].isnull().sum()


# In[296]:


dataset.drop("BLDSUGAR", axis=1, inplace=True)


# In[297]:


dataset["FEETCHK2"].value_counts()


# In[298]:


dataset["FEETCHK2"].isnull().sum()


# In[299]:


dataset.drop("FEETCHK2", axis=1, inplace=True)


# In[300]:


dataset["DOCTDIAB"].value_counts()


# In[301]:


dataset["DOCTDIAB"].isnull().sum()


# In[302]:


dataset.drop("DOCTDIAB", axis=1, inplace=True)


# In[303]:


dataset["CHKHEMO3"].value_counts()


# In[304]:


dataset.drop("CHKHEMO3", axis=1, inplace=True)


# In[305]:


dataset["FEETCHK"].value_counts()


# In[306]:


dataset.drop("FEETCHK", axis=1, inplace=True)


# In[307]:


dataset["EYEEXAM"].value_counts()


# In[308]:


dataset["EYEEXAM"].isnull().sum()


# In[309]:


dataset.drop("EYEEXAM", axis=1, inplace=True)


# In[310]:


dataset["DIABEYE"].value_counts()


# In[311]:


dataset.drop("DIABEYE", axis=1, inplace=True)


# In[312]:


dataset["DIABEDU"].value_counts()


# In[313]:


dataset.drop("DIABEDU", axis=1, inplace=True)


# In[314]:


dataset["CAREGIV1"].value_counts()


# In[315]:


dataset["CAREGIV1"].isnull().sum()


# In[316]:


dataset.drop("CAREGIV1", axis=1, inplace=True)


# In[317]:


dataset["CRGVREL1"].value_counts()


# In[318]:


dataset["CRGVREL1"].isnull().sum()


# In[319]:


dataset.drop("CRGVREL1", axis=1, inplace=True)


# In[320]:


dataset["CRGVLNG1"].value_counts()


# In[321]:


dataset["CRGVLNG1"].isnull().sum()


# In[322]:


dataset.drop("CRGVLNG1", axis=1, inplace=True)


# In[323]:


dataset["CRGVHRS1"].value_counts()


# In[324]:


dataset["CRGVHRS1"].isnull().sum()


# In[325]:


dataset.drop("CRGVHRS1", axis=1, inplace=True)


# In[326]:


dataset["CRGVPRB1"].value_counts()


# In[327]:


dataset["CRGVPRB1"].isnull().sum()


# In[328]:


dataset.drop("CRGVPRB1", axis=1, inplace=True)


# In[329]:


dataset["CRGVPERS"].value_counts()


# In[330]:


dataset.drop("CRGVPERS", axis=1, inplace=True)


# In[331]:


dataset["CRGVHOUS"].value_counts()


# In[332]:


dataset.drop("CRGVHOUS", axis=1, inplace=True)


# In[333]:


dataset["CRGVMST2"].value_counts()


# In[334]:


dataset.drop("CRGVMST2", axis=1, inplace=True)


# In[335]:


dataset["CRGVEXPT"].value_counts()


# In[336]:


dataset["CRGVEXPT"].isnull().sum()


# In[337]:


dataset.drop("CRGVEXPT", axis=1, inplace=True)


# In[338]:


dataset["CIMEMLOS"].value_counts()


# In[339]:


dataset["CIMEMLOS"].isnull().sum()


# In[340]:


dataset.drop("CIMEMLOS", axis=1, inplace=True)


# In[341]:


dataset["CDHOUSE"].value_counts()


# In[342]:


dataset.drop("CDHOUSE", axis=1, inplace=True)


# In[343]:


dataset["CDASSIST"].value_counts()


# In[344]:


dataset.drop("CDASSIST", axis=1, inplace=True)


# In[345]:


dataset["CDHELP"].value_counts()


# In[346]:


dataset.drop("CDHELP", axis=1, inplace=True)


# In[347]:


dataset["CDSOCIAL"].value_counts()


# In[348]:


dataset.drop("CDSOCIAL", axis=1, inplace=True)


# In[349]:


dataset["CDDISCUS"].value_counts()


# In[350]:


dataset.drop("CDDISCUS", axis=1, inplace=True)


# In[351]:


dataset["ARTTODAY"].value_counts()


# In[352]:


dataset.drop("ARTTODAY", axis=1, inplace=True)


# In[353]:


dataset["ARTHWGT"].value_counts()


# In[354]:


dataset.drop("ARTHWGT", axis=1, inplace=True)


# In[355]:


dataset["ARTHEXER"].value_counts()


# In[356]:


dataset.drop("ARTHEXER", axis=1, inplace=True)


# In[357]:


dataset["ARTHEDU"].value_counts()


# In[358]:


dataset.drop("ARTHEDU", axis=1, inplace=True)


# In[359]:


dataset["BLDSTOOL"].value_counts()


# In[360]:


dataset["BLDSTOOL"].isnull().sum()


# In[361]:


dataset.drop("BLDSTOOL", axis=1, inplace=True)


# In[362]:


dataset["LSTBLDS3"].value_counts()


# In[ ]:





# In[363]:


dataset.drop("LSTBLDS3", axis=1, inplace=True)


# In[364]:


dataset["HADSIGM3"].value_counts()


# In[365]:


dataset["HADSIGM3"].isnull().sum()


# In[366]:


dataset.drop("HADSIGM3", axis=1, inplace=True)


# In[367]:


dataset["HADSGCO1"].value_counts()


# In[368]:


dataset.drop("HADSGCO1", axis=1, inplace=True)


# In[369]:


dataset["LASTSIG3"].value_counts()


# In[370]:


dataset["LASTSIG3"].isnull().sum()


# In[371]:


dataset.drop("LASTSIG3", axis=1, inplace=True)


# In[372]:


dataset["TYPEWORK"].value_counts()


# In[373]:


dataset["TYPEWORK"].isnull().sum()


# In[374]:


dataset.drop("TYPEWORK",axis=1, inplace=True)


# In[375]:


dataset["TYPEINDS"].value_counts()


# In[376]:


dataset["TYPEINDS"].isnull().sum()


# In[377]:


dataset.drop("TYPEINDS", axis=1, inplace=True)


# In[378]:


dataset["SXORIENT"].value_counts()


# In[379]:


dataset["SXORIENT"].isnull().sum()


# In[380]:


dataset["TRNSGNDR"].value_counts()


# In[381]:


dataset["TRNSGNDR"].isnull().sum()


# In[382]:


dataset.drop("TRNSGNDR", axis=1, inplace=True)


# In[383]:


dataset["RCSBIRTH"].value_counts()


# In[384]:


dataset["RCSBIRTH"].isnull().sum()


# In[385]:


dataset.drop("RCSBIRTH", axis=1, inplace=True)


# In[386]:


dataset["RCSGENDR"].value_counts()


# In[387]:


dataset.drop("RCSGENDR", axis=1, inplace=True)


# In[388]:


dataset["RCHISLA1"].value_counts()


# In[389]:


dataset["RCHISLA1"].isnull().sum()


# In[390]:


dataset.drop("RCHISLA1", axis=1, inplace=True)


# In[391]:


dataset["RCSRACE1"].value_counts()


# In[392]:


dataset["RCSRACE1"].isnull().sum()


# In[393]:


dataset.drop("RCSRACE1", axis=1, inplace=True)


# In[394]:


dataset["RCSBRAC2"].value_counts()


# In[395]:


dataset.drop("RCSBRAC2", axis=1, inplace=True)


# In[396]:


dataset["RCSRLTN2"].value_counts()


# In[397]:


dataset["RCSRLTN2"].isnull().sum()


# In[398]:


dataset.drop("RCSRLTN2", axis=1, inplace=True)


# In[399]:


dataset["CASTHDX2"].value_counts()


# In[400]:


dataset.drop("CASTHDX2", axis=1, inplace=True)


# In[401]:


dataset["CASTHNO2"].value_counts()


# In[402]:


dataset.drop("CASTHNO2", axis=1, inplace=True)


# In[403]:


dataset["ADHISPA"].value_counts()


# In[404]:


dataset.drop("ADHISPA", axis=1, inplace=True)


# In[405]:


dataset["CHHISPA"].value_counts()


# In[406]:


dataset.drop("CHHISPA", axis=1, inplace=True)


# In[407]:


dataset["QSTVER"].value_counts()


# In[408]:


dataset["QSTVER"].isnull().sum()


# In[409]:


dataset["QSTLANG"].value_counts()


# In[410]:


dataset["QSTLANG"].isnull().sum()


# In[411]:


dataset["EXACTOT1"].value_counts()


# In[412]:


dataset["EXACTOT1"].isnull().sum()


# In[413]:


dataset["EXACTOT2"].value_counts()


# In[414]:


dataset["EXACTOT2"].isnull().sum()


# In[415]:


dataset["_MSACODE"].value_counts()


# In[416]:


dataset.drop("_MSACODE", axis=1, inplace=True)


# In[417]:


dataset["MSCODE"].value_counts()


# In[418]:


dataset.drop("MSCODE", axis=1, inplace=True)


# In[419]:


dataset["_STSTR"].value_counts()


# In[420]:


dataset["_STSTR"].isnull().sum()


# In[421]:


dataset["_STRWT"].value_counts()


# In[422]:


dataset["_STRWT"].isnull().sum()


# In[423]:


dataset["_RAW"].value_counts()


# In[424]:


dataset["_RAW"].isnull().sum()


# In[425]:


dataset["_WT2"].value_counts()


# In[426]:


dataset["_WT2"].isnull().sum()


# In[427]:


dataset["_RAWRAKE"].value_counts()


# In[428]:


dataset["_RAWRAKE"].isnull().sum()


# In[429]:


dataset["_WT2RAKE"].value_counts()


# In[430]:


dataset["_WT2RAKE"].isnull().sum()


# In[431]:


dataset["_REGION"].value_counts()


# In[432]:


dataset["_REGION"].isnull().sum()


# In[433]:


dataset["_IMPAGE"].value_counts()


# In[434]:


dataset["_IMPAGE"].isnull().sum()


# In[435]:


dataset["_IMPRACE"].value_counts()


# In[436]:


dataset.drop("_IMPRACE", axis=1, inplace=True)


# In[437]:


dataset["_IMPNPH"].value_counts()


# In[438]:


dataset["_IMPNPH"].isnull().sum()


# In[439]:


dataset["_IMPEDUC"].value_counts()


# In[440]:


dataset["_IMPEDUC"].isnull().sum()


# In[441]:


dataset["_IMPMRTL"].value_counts()


# In[442]:


dataset["_IMPMRTL"].isnull().sum()


# In[443]:


dataset["_IMPHOME"].value_counts()


# In[444]:


dataset["_IMPHOME"].isnull().sum()


# In[445]:


dataset["O_STATE"].value_counts()


# In[446]:


dataset["O_STATE"].isnull().sum()


# In[447]:


dataset["_CHISPNC"].value_counts()


# In[448]:


dataset["_CHISPNC"].isnull().sum()


# In[449]:


dataset.drop("_CHISPNC", axis=1, inplace=True)


# In[450]:


dataset["_CRACE1"].value_counts()


# In[451]:


dataset.drop("_CRACE1", axis=1, inplace=True)


# In[452]:


dataset["_CPRACE"].value_counts()


# In[453]:


dataset.drop("_CPRACE", axis=1, inplace=True)


# In[454]:


dataset["_IMPCAGE"].value_counts()


# In[455]:


dataset.drop("_IMPCAGE", axis=1, inplace=True)


# In[456]:


dataset["_IMPCRAC"].value_counts()


# In[457]:


dataset.drop("_IMPCRAC", axis=1, inplace=True)


# In[458]:


dataset["_IMPCSEX"].value_counts()


# In[459]:


dataset.drop("_IMPCSEX", axis=1, inplace=True)


# In[460]:


dataset["_RAWCH"].value_counts()


# In[461]:


dataset["_RAWCH"].isnull().sum()


# In[462]:


dataset.drop("_RAWCH", axis=1, inplace=True)


# In[463]:


dataset["_WT2CH"].value_counts()


# In[464]:


dataset["_WT2CH"].isnull().sum()


# In[465]:


dataset.drop("_WT2CH", axis=1, inplace=True)


# In[466]:


dataset["_CLCM1V1"].value_counts()


# In[467]:


dataset["_CLCM1V1"].isnull().sum()


# In[468]:


dataset.drop("_CLCM1V1", axis=1, inplace=True)


# In[469]:


dataset["_CLCM2V1"].value_counts()


# In[470]:


dataset.drop("_CLCM2V1", axis=1, inplace=True)


# In[471]:


dataset["_CLCM3V1"].value_counts()


# In[472]:


dataset.drop("_CLCM3V1", axis=1, inplace=True)


# In[473]:


dataset["_CLCM4V1"].value_counts()


# In[474]:


dataset["_CLCM4V1"].isnull().sum()


# In[475]:


dataset.drop("_CLCM4V1", axis=1, inplace=True)


# In[476]:


dataset["_CLCM5V1"].value_counts()


# In[477]:


dataset.drop("_CLCM5V1", axis=1, inplace=True)


# In[478]:


dataset["_CLCWTV1"].value_counts()


# In[479]:


dataset["_CLCWTV1"].isnull().sum()


# In[480]:


dataset.drop("_CLCWTV1", axis=1, inplace=True)


# In[481]:


dataset["_DUALUSE"].value_counts()


# In[482]:


dataset["_DUALUSE"].isnull().sum()


# In[483]:


dataset["_DUALCOR"].value_counts()


# In[484]:


dataset["_DUALCOR"].isnull().sum()


# In[485]:


dataset["_LLCPM01"].value_counts()


# In[486]:


dataset["_LLCPM01"].isnull().sum()


# In[487]:


dataset["_LLCPM02"].value_counts()


# In[488]:


dataset["_LLCPM02"].isnull().sum()


# In[489]:


dataset["_LLCPM03"].value_counts()


# In[490]:


dataset["_LLCPM03"].isnull().sum()


# In[491]:


dataset["_LLCPM04"].value_counts()


# In[492]:


dataset["_LLCPM04"].isnull().sum()


# In[493]:


dataset["_LLCPM05"].value_counts()


# In[494]:


dataset["_LLCPM05"].isnull().sum()


# In[495]:


dataset["_LLCPM06"].value_counts()


# In[496]:


dataset["_LLCPM06"].isnull().sum()


# In[497]:


dataset["_LLCPM07"].value_counts()


# In[498]:


dataset["_LLCPM07"].isnull().sum()


# In[499]:


dataset["_LLCPM08"].value_counts()


# In[500]:


dataset["_LLCPM08"].isnull().sum()


# In[501]:


dataset["_LLCPM09"].value_counts()


# In[502]:


dataset["_LLCPM09"].isnull().sum()


# In[503]:


dataset["_LLCPM10"].value_counts()


# In[504]:


dataset["_LLCPM10"].isnull().sum()


# In[505]:


dataset["_LLCPM11"].isnull().sum()


# In[506]:


dataset["_LLCPM12"].value_counts()


# In[507]:


dataset["_LLCPM12"].isnull().sum()


# In[508]:


dataset["_LLCPM13"].value_counts()


# In[509]:


dataset["_LLCPM13"].isnull().sum()


# In[510]:


dataset["_LLCPM14"].value_counts()


# In[511]:


dataset["_LLCPM14"].isnull().sum()


# In[512]:


dataset["_LLCPM15"].value_counts()


# In[513]:


dataset["_LLCPM15"].isnull().sum()


# In[514]:


dataset["_LLCPM16"].isnull().sum()


# In[515]:


dataset["_LLCPWT"].value_counts()


# In[516]:


dataset["_LLCPWT"].isnull().sum()


# In[517]:


dataset["_LCM01V1"].value_counts()


# In[518]:


dataset["_LCM01V1"].isnull().sum()


# In[519]:


dataset.drop("_LCM01V1", axis=1, inplace=True)


# In[520]:


dataset["_LCM02V1"].value_counts()


# In[521]:


dataset["_LCM02V1"].isnull().sum()


# In[522]:


dataset.drop("_LCM02V1", axis=1, inplace=True)


# In[523]:


dataset["_LCM03V1"].value_counts()


# In[524]:


dataset["_LCM03V1"].isnull().sum()


# In[525]:


dataset.drop("_LCM03V1", axis=1, inplace=True)


# In[526]:


dataset["_LCM04V1"].value_counts()


# In[527]:


dataset["_LCM04V1"].isnull().sum()


# In[528]:


dataset.drop("_LCM04V1", axis=1, inplace=True)


# In[529]:


dataset["_LCM05V1"].value_counts()


# In[530]:


dataset["_LCM05V1"].isnull().sum()


# In[531]:


dataset.drop("_LCM05V1", axis=1, inplace=True)


# In[532]:


dataset["_LCM06V1"].value_counts()


# In[533]:


dataset["_LCM06V1"].isnull().sum()


# In[534]:


dataset.drop("_LCM06V1", axis=1, inplace=True)


# In[535]:


dataset["_LCM07V1"].value_counts()


# In[536]:


dataset["_LCM07V1"].isnull().sum()


# In[537]:


dataset.drop("_LCM07V1", axis=1, inplace=True)


# In[538]:


dataset["_LCM08V1"].value_counts()


# In[539]:


dataset["_LCM08V1"].isnull().sum()


# In[540]:


dataset.drop("_LCM08V1", axis=1, inplace=True)


# In[541]:


dataset["_LCPWTV1"].value_counts()


# In[542]:


dataset["_LCPWTV1"].isnull().sum()


# In[543]:


dataset.drop("_LCPWTV1", axis=1, inplace=True)


# In[544]:


dataset["_LCM01V2"].value_counts()


# In[545]:


dataset["_LCM01V2"].isnull().sum()


# In[546]:


dataset.drop("_LCM01V2", axis=1, inplace=True)


# In[547]:


dataset["_LCM02V2"].value_counts()


# In[548]:


dataset["_LCM02V2"].isnull().sum()


# In[549]:


dataset.drop("_LCM02V2", axis=1, inplace=True)


# In[550]:


dataset["_LCM03V2"].value_counts()


# In[551]:


dataset["_LCM03V2"].isnull().sum()


# In[552]:


dataset.drop("_LCM03V2", axis=1, inplace=True)


# In[553]:


dataset["_LCM04V2"].value_counts()


# In[554]:


dataset["_LCM04V2"].isnull().sum()


# In[555]:


dataset.drop("_LCM04V2", axis=1, inplace=True)


# In[556]:


dataset["_LCM05V2"].value_counts()


# In[557]:


dataset["_LCM05V2"].isnull().sum()


# In[558]:


dataset.drop("_LCM05V2", axis=1, inplace=True)


# In[559]:


dataset["_LCM06V2"].value_counts()


# In[560]:


dataset["_LCM06V2"].isnull().sum()


# In[561]:


dataset.drop("_LCM06V2", axis=1, inplace=True)


# In[562]:


dataset["_LCM07V2"].value_counts()


# In[563]:


dataset["_LCM07V2"].isnull().sum()


# In[564]:


dataset.drop("_LCM07V2", axis=1, inplace=True)


# In[565]:


dataset["_LCM08V2"].value_counts()


# In[566]:


dataset["_LCM08V2"].isnull().sum()


# In[567]:


dataset.drop("_LCM08V2", axis=1, inplace=True)


# In[568]:


dataset["_LCM09V2"].value_counts()


# In[569]:


dataset["_LCM09V2"].isnull().sum()


# In[570]:


dataset.drop("_LCM09V2", axis=1, inplace=True)


# In[571]:


dataset["_LCM10V2"].value_counts()


# In[572]:


dataset["_LCM10V2"].isnull().sum()


# In[573]:


dataset.drop("_LCM10V2", axis=1, inplace=True)


# In[574]:


dataset["_LCM11V2"].value_counts()


# In[575]:


dataset["_LCM11V2"].isnull().sum()


# In[576]:


dataset.drop("_LCM11V2", axis=1, inplace=True)


# In[577]:


dataset["_LCM12V2"].value_counts()


# In[578]:


dataset["_LCM12V2"].isnull().sum()


# In[579]:


dataset.drop("_LCM12V2", axis=1, inplace=True)


# In[580]:


dataset["_LCPWTV2"].isnull().sum()


# In[581]:


dataset.drop("_LCPWTV2", axis=1, inplace=True)


# In[582]:


dataset["_RFHLTH"].value_counts()


# In[583]:


dataset["_RFHLTH"].isnull().sum()


# In[584]:


dataset["_HCVU651"].value_counts()


# In[585]:


dataset["_HCVU651"].isnull().sum()


# In[586]:


dataset["_RFHYPE5"].value_counts()


# In[587]:


dataset["_RFHYPE5"].isnull().sum()


# In[588]:


dataset["_CHOLCHK"].value_counts()


# In[589]:


dataset["_CHOLCHK"].isnull().sum()


# In[590]:


dataset["_RFCHOL"].value_counts()


# In[591]:


dataset["_RFCHOL"].isnull().sum()


# In[592]:


dataset["_MICHD"].value_counts()


# In[593]:


dataset["_MICHD"].isnull().sum()


# In[594]:


dataset["_LTASTH1"].value_counts()


# In[595]:


dataset["_LTASTH1"].isnull().sum()


# In[596]:


dataset["_CASTHM1"].value_counts()


# In[597]:


dataset["_CASTHM1"].isnull().sum()


# In[598]:


dataset["_ASTHMS1"].value_counts()


# In[599]:


dataset["_ASTHMS1"].isnull().sum()


# In[600]:


dataset["_DRDXAR1"].value_counts()


# In[601]:


dataset["_DRDXAR1"].isnull().sum()


# In[602]:


dataset["_MRACE1"].value_counts()


# In[603]:


dataset["_MRACE1"].isnull().sum()


# In[604]:


dataset.drop("_MRACE1", axis=1, inplace=True)


# In[605]:


dataset["_M_RACE"].value_counts()


# In[606]:


dataset["_M_RACE"].isnull().sum()


# In[607]:


dataset.drop("_M_RACE", axis=1, inplace=True)


# In[608]:


dataset["_HISPANC"].value_counts()


# In[609]:


dataset["_HISPANC"].isnull().sum()


# In[610]:


dataset.drop("_HISPANC", axis=1, inplace=True)


# In[611]:


dataset["_RACE"].value_counts()


# In[612]:


dataset["_RACE"].isnull().sum()


# In[613]:


dataset.drop("_RACE", axis=1, inplace=True)


# In[614]:


dataset["_RACEG21"].value_counts()


# In[615]:


dataset["_RACEG21"].isnull().sum()


# In[616]:


dataset.drop("_RACEG21", axis=1, inplace=True)


# In[617]:


dataset["_RACEGR3"].value_counts()


# In[618]:


dataset["_RACEGR3"].isnull().sum()


# In[619]:


dataset.drop("_RACEGR3", axis=1, inplace=True)


# In[620]:


dataset["_RACE_G1"].value_counts()


# In[621]:


dataset["_RACE_G1"].isnull().sum()


# In[622]:


dataset.drop("_RACE_G1", axis=1, inplace=True)


# In[623]:


dataset["_AGEG5YR"].value_counts()


# In[624]:


dataset["_AGEG5YR"].isnull().sum()


# In[625]:


dataset["_AGE65YR"].value_counts()


# In[626]:


dataset["_AGE65YR"].isnull().sum()


# In[627]:


dataset["_AGE80"].value_counts()


# In[628]:


dataset["_AGE80"].isnull().sum()


# In[629]:


dataset["_AGE_G"].value_counts()


# In[630]:


dataset["_AGE_G"].isnull().sum()


# In[631]:


dataset["HTIN4"].value_counts()


# In[632]:


dataset["HTIN4"].isnull().sum()


# In[633]:


dataset.drop("HTIN4", axis=1, inplace=True)


# In[634]:


dataset["HTM4"].value_counts()


# In[635]:


dataset["HTM4"].isnull().sum()


# In[636]:


dataset.drop("HTM4", axis=1, inplace=True)


# In[637]:


dataset["WTKG3"].value_counts()


# In[638]:


dataset["WTKG3"].isnull().sum()


# In[639]:


dataset.drop("WTKG3", axis=1, inplace=True)


# In[640]:


dataset["_BMI5"].value_counts()


# In[641]:


dataset["_BMI5"].isnull().sum()


# In[642]:


dataset["_BMI5CAT"].value_counts()


# In[643]:


dataset["_BMI5CAT"].isnull().sum()


# In[644]:


dataset["_RFBMI5"].value_counts()


# In[645]:


dataset["_RFBMI5"].isnull().sum()


# In[646]:


dataset["_CHLDCNT"].value_counts()


# In[647]:


dataset["_CHLDCNT"].isnull().sum()


# In[648]:


dataset["_EDUCAG"].value_counts()


# In[649]:


dataset["_EDUCAG"].isnull().sum()


# In[650]:


dataset["_INCOMG"].value_counts()


# In[651]:


dataset["_INCOMG"].isnull().sum()


# In[652]:


dataset["_SMOKER3"].value_counts()


# In[653]:


dataset["_SMOKER3"].isnull().sum()


# In[654]:


dataset["_RFSMOK3"].value_counts()


# In[655]:


dataset["_RFSMOK3"].isnull().sum()


# In[656]:


dataset["DRNKANY5"].value_counts()


# In[657]:


dataset["DRNKANY5"].isnull().sum()


# In[658]:


dataset["DROCDY3_"].value_counts()


# In[659]:


dataset["DROCDY3_"].isnull().sum()


# In[660]:


dataset["_RFBING5"].value_counts()


# In[661]:


dataset["_RFBING5"].isnull().sum()


# In[662]:


dataset["_DRNKWEK"].value_counts()


# In[663]:


dataset["_DRNKWEK"].isnull().sum()


# In[664]:


dataset["_RFDRHV5"].value_counts()


# In[665]:


dataset["_RFDRHV5"].isnull().sum()


# In[666]:


dataset["FTJUDA1_"].value_counts()


# In[667]:


dataset["FTJUDA1_"].isnull().sum()


# In[668]:


dataset["FRUTDA1_"].value_counts()


# In[669]:


dataset["FRUTDA1_"].isnull().sum()


# In[670]:


dataset["BEANDAY_"].value_counts()


# In[671]:


dataset["BEANDAY_"].isnull().sum()


# In[672]:


dataset["GRENDAY_"].value_counts()


# In[673]:


dataset["GRENDAY_"].isnull().sum()


# In[674]:


dataset["ORNGDAY_"].value_counts()


# In[675]:


dataset["ORNGDAY_"].isnull().sum()


# In[676]:


dataset["VEGEDA1_"].value_counts()


# In[677]:


dataset["VEGEDA1_"].isnull().sum()


# In[678]:


dataset["_MISFRTN"].value_counts()


# In[679]:


dataset["_MISFRTN"].isnull().sum()


# In[680]:


dataset["_MISVEGN"].value_counts()


# In[681]:


dataset["_MISVEGN"].isnull().sum()


# In[682]:


dataset["_FRTRESP"].value_counts()


# In[683]:


dataset["_FRTRESP"].isnull().sum()


# In[684]:


dataset["_VEGRESP"].value_counts()


# In[685]:


dataset["_VEGRESP"].isnull().sum()


# In[686]:


dataset["_FRUTSUM"].value_counts()


# In[687]:


dataset["_FRUTSUM"].isnull().sum()


# In[688]:


dataset["_VEGESUM"].value_counts()


# In[689]:


dataset["_VEGESUM"].isnull().sum()


# In[690]:


dataset["_FRTLT1"].value_counts()


# In[691]:


dataset["_FRTLT1"].isnull().sum()


# In[692]:


dataset["_VEGLT1"].value_counts()


# In[693]:


dataset["_VEGLT1"].isnull().sum()


# In[694]:


dataset["_FRT16"].value_counts()


# In[695]:


dataset["_FRT16"].isnull().sum()


# In[696]:


dataset["_VEG23"].value_counts()


# In[697]:


dataset["_VEG23"].isnull().sum()


# In[698]:


dataset["_FRUITEX"].value_counts()


# In[699]:


dataset["_FRUITEX"].isnull().sum()


# In[700]:


dataset["_VEGETEX"].value_counts()


# In[701]:


dataset["_VEGETEX"].isnull().sum()


# In[702]:


dataset["_TOTINDA"].value_counts()


# In[703]:


dataset["_TOTINDA"].isnull().sum()


# In[704]:


dataset["METVL11_"].value_counts()


# In[705]:


dataset["METVL11_"].isnull().sum()


# In[706]:


dataset["METVL21_"].value_counts()


# In[707]:


dataset["METVL21_"].isnull().sum()


# In[708]:


dataset["MAXVO2_"].value_counts()


# In[709]:


dataset["MAXVO2_"].isnull().sum()


# In[710]:


dataset["FC60_"].value_counts()


# In[711]:


dataset["FC60_"].isnull().sum()


# In[712]:


dataset["ACTIN11_"].value_counts()


# In[713]:


dataset["ACTIN11_"].isnull().sum()


# In[714]:


dataset["ACTIN21_"].value_counts()


# In[715]:


dataset["ACTIN21_"].isnull().sum()


# In[716]:


dataset["PADUR1_"].value_counts()


# In[717]:


dataset["PADUR1_"].isnull().sum()


# In[718]:


dataset["PADUR2_"].value_counts()


# In[719]:


dataset["PADUR2_"].isnull().sum()


# In[720]:


dataset.drop("PADUR2_", axis=1, inplace=True)


# In[721]:


dataset["PAFREQ1_"].value_counts()


# In[722]:


dataset["PAFREQ1_"].isnull().sum()


# In[723]:


dataset["PAFREQ2_"].value_counts()


# In[724]:


dataset["PAFREQ2_"].isnull().sum()


# In[725]:


dataset.drop("PAFREQ2_", axis=1, inplace=True)


# In[726]:


dataset["_MINAC11"].value_counts()


# In[727]:


dataset["_MINAC11"].isnull().sum()


# In[728]:


dataset["_MINAC21"].value_counts()


# In[729]:


dataset["_MINAC21"].isnull().sum()


# In[730]:


dataset["STRFREQ_"].value_counts()


# In[731]:


dataset["STRFREQ_"].isnull().sum()


# In[732]:


dataset["PAMISS1_"].value_counts()


# In[733]:


dataset["PAMISS1_"].isnull().sum()


# In[734]:


dataset["PAMIN11_"].value_counts()


# In[735]:


dataset["PAMIN11_"].isnull().sum()


# In[736]:


dataset["PAMIN21_"].value_counts()


# In[737]:


dataset["PAMIN21_"].isnull().sum()


# In[738]:


dataset["PA1MIN_"].value_counts()


# In[739]:


dataset["PA1MIN_"].isnull().sum()


# In[740]:


dataset["PAVIG11_"].value_counts()


# In[741]:


dataset["PAVIG11_"].isnull().sum()


# In[742]:


dataset["PAVIG21_"].value_counts()


# In[743]:


dataset["PAVIG21_"].isnull().sum()


# In[744]:


dataset["PA1VIGM_"].value_counts()


# In[745]:


dataset["PA1VIGM_"].isnull().sum()


# In[746]:


dataset["_PACAT1"].value_counts()


# In[747]:


dataset["_PACAT1"].isnull().sum()


# In[748]:


dataset["_PAINDX1"].value_counts()


# In[749]:


dataset["_PAINDX1"].isnull().sum()


# In[750]:


dataset["_PA150R2"].value_counts()


# In[751]:


dataset["_PA150R2"].isnull().sum()


# In[752]:


dataset["_PA300R2"].value_counts()


# In[753]:


dataset["_PA300R2"].isnull().sum()


# In[754]:


dataset["_PA30021"].value_counts()


# In[755]:


dataset["_PA30021"].isnull().sum()


# In[756]:


dataset["_PASTRNG"].value_counts()


# In[757]:


dataset["_PASTRNG"].isnull().sum()


# In[758]:


dataset["_PAREC1"].value_counts()


# In[759]:


dataset["_PAREC1"].isnull().sum()


# In[760]:


dataset["_PASTAE1"].value_counts()


# In[761]:


dataset["_PASTAE1"].isnull().sum()


# In[762]:


dataset["_LMTACT1"].value_counts()


# In[763]:


dataset["_LMTACT1"].isnull().sum()


# In[764]:


dataset["_LMTWRK1"].value_counts()


# In[765]:


dataset["_LMTWRK1"].isnull().sum()


# In[766]:


dataset["_LMTSCL1"].value_counts()


# In[767]:


dataset["_LMTSCL1"].isnull().sum()


# In[768]:


dataset["_RFSEAT2"].value_counts()


# In[769]:


dataset["_RFSEAT2"].isnull().sum()


# In[770]:


dataset["_RFSEAT3"].value_counts()


# In[771]:


dataset["_RFSEAT3"].isnull().sum()


# In[772]:


dataset["_FLSHOT6"].value_counts()


# In[773]:


dataset["_FLSHOT6"].isnull().sum()


# In[774]:


dataset["_PNEUMO2"].value_counts()


# In[775]:


dataset["_PNEUMO2"].isnull().sum()


# In[776]:


dataset["_AIDTST3"].value_counts()


# In[777]:


dataset["_AIDTST3"].isnull().sum()


# In[778]:


dataset["MEDICARE"].value_counts()


# In[779]:


dataset["MEDICARE"].isnull().sum()


# In[780]:


dataset["HLTHCVR1"].value_counts()


# In[781]:


dataset["HLTHCVR1"].isnull().sum()


# In[782]:


dataset["DELAYMED"].value_counts()


# In[783]:


dataset["DELAYMED"].isnull().sum()


# In[784]:


dataset["NOCOV121"].value_counts()


# In[785]:


dataset["NOCOV121"].isnull().sum()


# In[786]:


dataset["LSTCOVRG"].value_counts()


# In[787]:


dataset["LSTCOVRG"].isnull().sum()


# In[788]:


dataset.drop("LSTCOVRG", axis=1, inplace=True)


# In[789]:


dataset["DRVISITS"].value_counts()


# In[790]:


dataset["DRVISITS"].isnull().sum()


# In[791]:


dataset["MEDCOST"].value_counts()


# In[792]:


dataset["MEDSCOST"].isnull().sum()


# In[793]:


dataset["CARERCVD"].value_counts()


# In[794]:


dataset["CARERCVD"].isnull().sum()


# In[795]:


dataset["MEDBILL1"].value_counts()


# In[796]:


dataset["MEDBILL1"].isnull().sum()


# In[797]:


dataset["EMPLSTYR"].value_counts()


# In[798]:


dataset["EMPLSTYR"].isnull().sum()


# In[799]:


dataset.drop("EMPLSTYR", axis=1, inplace=True)


# In[800]:


dataset["JOBINJMT"].value_counts()


# In[801]:


dataset["JOBINJMT"].isnull().sum()


# In[802]:


dataset.drop("JOBINJMT", axis=1, inplace=True)


# In[803]:


dataset["DAYSRTRN"].value_counts()


# In[804]:


dataset["DAYSRTRN"].isnull().sum()


# In[805]:


dataset.drop("DAYSRTRN", axis=1, inplace=True)


# In[806]:


dataset["WHOPAIDT"].value_counts()


# In[807]:


dataset["WHOPAIDT"].isnull().sum()


# In[808]:


dataset.drop("WHOPAIDT", axis=1, inplace=True)


# In[809]:


dataset["OTHRPAID"].value_counts()


# In[810]:


dataset["OTHRPAID"].isnull().sum()


# In[811]:


dataset.drop("OTHRPAID", axis=1, inplace=True)


# In[812]:


dataset["EMPAWARE"].value_counts()


# In[813]:


dataset["EMPAWARE"].isnull().sum()


# In[814]:


dataset.drop("EMPAWARE", axis=1, inplace=True)


# In[815]:


dataset["MISNERVS"].value_counts()


# In[816]:


dataset["MISNERVS"].isnull().sum()


# In[817]:


dataset.drop("MISNERVS", axis=1, inplace=True)


# In[818]:


dataset["MISHOPLS"].value_counts()


# In[819]:


dataset["MISHOPLS"].isnull().sum()


# In[820]:


dataset.drop("MISHOPLS", axis=1, inplace=True)


# In[821]:


dataset["MISRSTLS"].value_counts()


# In[822]:


dataset["MISRSTLS"].isnull().sum()


# In[823]:


dataset.drop("MISRSTLS", axis=1, inplace=True)


# In[824]:


dataset["MISDEPRD"].value_counts()


# In[825]:


dataset["MISDEPRD"].isnull().sum()


# In[826]:


dataset.drop("MISDEPRD", axis=1, inplace=True)


# In[827]:


dataset["MISEFFRT"].value_counts()


# In[828]:


dataset["MISEFFRT"].isnull().sum()


# In[829]:


dataset.drop("MISEFFRT", axis=1, inplace=True)


# In[830]:


dataset["MISWTLES"].value_counts()


# In[831]:


dataset["MISWTLES"].isnull().sum()


# In[832]:


dataset.drop("MISWTLES", axis=1, inplace=True)


# In[833]:


dataset["MISNOWRK"].value_counts()


# In[834]:


dataset["MISNOWRK"].isnull().sum()


# In[835]:


dataset.drop("MISNOWRK", axis=1, inplace=True)


# In[836]:


dataset["MISTMNT"].value_counts()


# In[837]:


dataset["MISTMNT"].isnull().sum()


# In[838]:


dataset.drop("MISTMNT", axis=1, inplace=True)


# In[839]:


dataset["MISTRHLP"].value_counts()


# In[840]:


dataset["MISTRHLP"].isnull().sum()


# In[841]:


dataset.drop("MISTRHLP", axis=1, inplace=True)


# In[842]:


dataset["MISPHLPF"].value_counts()


# In[843]:


dataset["MISPHLPF"].isnull().sum()


# In[844]:


dataset.drop("MISPHLPF", axis=1, inplace=True)


# In[845]:


dataset["SSBSUGR1"].value_counts()


# In[846]:


dataset["SSBSUGR1"].isnull().sum()


# In[847]:


dataset.drop("SSBSUGR1", axis=1, inplace=True)


# In[848]:


dataset["SSBFRUT2"].value_counts()


# In[849]:


dataset["SSBFRUT2"].isnull().sum()


# In[850]:


dataset.drop("SSBFRUT2", axis=1, inplace=True)


# In[851]:


dataset["HCVHEAR"].value_counts()


# In[852]:


dataset["HCVHEAR"].isnull().sum()


# In[853]:


dataset.drop("HCVHEAR", axis=1, inplace=True)


# In[854]:


dataset["HCVTEST"].value_counts()


# In[855]:


dataset["HCVTEST"].isnull().sum()


# In[856]:


dataset.drop("HCVTEST", axis=1, inplace=True)


# In[857]:


dataset["HCVLASTT"].value_counts()


# In[858]:


dataset["HCVLASTT"].isnull().sum()


# In[859]:


dataset.drop("HCVLASTT", axis=1, inplace=True)


# In[860]:


dataset["HCVINPTR"].value_counts()


# In[861]:


dataset["HCVINPTR"].isnull().sum()


# In[862]:


dataset.drop("HCVINPTR", axis=1, inplace=True)


# In[863]:


dataset["HCVINPTO"].value_counts()


# In[864]:


dataset["HCVINPTO"].isnull().sum()


# In[865]:


dataset.drop("HCVINPTO", axis=1, inplace=True)


# In[866]:


dataset["HCVINPTA"].value_counts()


# In[867]:


dataset["HCVINPTA"].isnull().sum()


# In[868]:


dataset.drop("HCVINPTA", axis=1, inplace=True)


# In[869]:


dataset["HCVPRIMR"].value_counts()


# In[870]:


dataset["HCVPRIMR"].isnull().sum()


# In[871]:


dataset.drop("HCVPRIMR", axis=1, inplace=True)


# In[872]:


dataset["HCVPRIMO"].value_counts()


# In[873]:


dataset["HCVPRIMO"].isnull().sum()


# In[874]:


dataset.drop("HCVPRIMO", axis=1, inplace=True)


# In[875]:


dataset["HCVPRIMA"].value_counts()


# In[876]:


dataset["HCVPRIMA"].isnull().sum()


# In[877]:


dataset.drop("HCVPRIMA", axis=1, inplace=True)


# In[878]:


dataset["HEALTHCL1"].value_counts()


# In[879]:


dataset["HEALTHCL1"].isnull().sum()


# In[880]:


dataset.drop("HEALTHCL1", axis=1, inplace=True)


# In[881]:


dataset["LIFECHG"].value_counts()


# In[882]:


dataset["LIFECHG"].isnull().sum()


# In[883]:


dataset.drop("LIFECHG", axis=1, inplace=True)


# In[884]:


dataset["LASTDENT1"].value_counts()


# In[885]:


dataset["LASTDENT1"].isnull().sum()


# In[886]:


dataset.drop("LASTDENT1", axis=1, inplace=True)


# In[887]:


dataset["RMVTEETH1"].value_counts()


# In[888]:


dataset["RMVTEETH1"].isnull().sum()


# In[889]:


dataset.drop("RMVTEETH1", axis=1, inplace=True)


# In[890]:


dataset["DIFFHEAR"].value_counts()


# In[891]:


dataset["DIFFHEAR"].isnull().sum()


# In[892]:


dataset.drop("DIFFHEAR", axis=1, inplace=True)


# In[893]:


dataset["FRUITVEG"].value_counts()


# In[894]:


dataset["FRUITVEG"].isnull().sum()


# In[895]:


dataset.drop("FRUITVEG", axis=1, inplace=True)


# In[896]:


dataset["NOVEGFRU"].value_counts()


# In[897]:


dataset["NOVEGFRU"].isnull().sum()


# In[898]:


dataset.drop("NOVEGFRU", axis=1, inplace=True)


# In[899]:


dataset["NOVFOTHR"].value_counts()


# In[900]:


dataset["NOVFOTHR"].isnull().sum()


# In[901]:


dataset.drop("NOVFOTHR", axis=1, inplace=True)


# In[902]:


dataset["STRSRENT"].value_counts()


# In[903]:


dataset["STRSRENT"].isnull().sum()


# In[904]:


dataset.drop("STRSRENT", axis=1, inplace=True)


# In[905]:


dataset["STRSMEAL"].value_counts()


# In[906]:


dataset["STRSMEAL"].isnull().sum()


# In[907]:


dataset.drop("STRSMEAL", axis=1, inplace=True)


# In[908]:


dataset["dsripreg"].value_counts()


# In[909]:


dataset["dsripreg"].isnull().sum()


# In[910]:


dataset["REGION"].value_counts()


# In[911]:


dataset["REGION"].isnull().sum()


# In[912]:


dataset["PPS_1"].value_counts()


# In[913]:


dataset["PPS_1"].isnull().sum()


# In[914]:


dataset["PPS_3"].value_counts()


# In[915]:


dataset["PPS_3"].isnull().sum()


# In[916]:


dataset["PPS_8"].value_counts()


# In[917]:


dataset["PPS_8"].isnull().sum()


# In[918]:


dataset["PPS_9"].value_counts()


# In[919]:


dataset["PPS_9"].isnull().sum()


# In[920]:


dataset["PPS_14"].value_counts()


# In[921]:


dataset["PPS_14"].isnull().sum()


# In[922]:


dataset["PPS_16"].value_counts()


# In[923]:


dataset["PPS_16"].isnull().sum()


# In[924]:


dataset["PPS_19"].value_counts()


# In[925]:


dataset["PPS_19"].isnull().sum()


# In[926]:


dataset["PPS_20"].value_counts()


# In[927]:


dataset["PPS_20"].isnull().sum()


# In[928]:


dataset["PPS_21"].value_counts()


# In[929]:


dataset["PPS_21"].isnull().sum()


# In[930]:


dataset["PPS_22"].value_counts()


# In[931]:


dataset["PPS_22"].isnull().sum()


# In[932]:


dataset["PPS_23"].value_counts()


# In[933]:


dataset["PPS_23"].isnull().sum()


# In[934]:


dataset["PPS_25"].value_counts()


# In[935]:


dataset["PPS_25"].isnull().sum()


# In[936]:


dataset["PPS_27"].value_counts()


# In[937]:


dataset["PPS_27"].isnull().sum()


# In[938]:


dataset["PPS_32"].value_counts()


# In[939]:


dataset["PPS_32"].isnull().sum()


# In[940]:


dataset["PPS_33"].value_counts()


# In[941]:


dataset["PPS_33"].isnull().sum()


# In[942]:


dataset["PPS_34"].value_counts()


# In[943]:


dataset["PPS_34"].isnull().sum()


# In[944]:


dataset["PPS_36"].value_counts()


# In[945]:


dataset["PPS_36"].isnull().sum()


# In[946]:


dataset["PPS_39"].value_counts()


# In[947]:


dataset["PPS_39"].isnull().sum()


# In[948]:


dataset["PPS_40"].value_counts()


# In[949]:


dataset["PPS_40"].isnull().sum()


# In[950]:


dataset["PPS_43"].value_counts()


# In[951]:


dataset["PPS_43"].isnull().sum()


# In[952]:


dataset["PPS_44"].value_counts()


# In[953]:


dataset["PPS_44"].isnull().sum()


# In[954]:


dataset["PPS_45"].value_counts()


# In[955]:


dataset["PPS_45"].isnull().sum()


# In[956]:


dataset["PPS_46"].value_counts()


# In[957]:


dataset["PPS_46"].isnull().sum()


# In[958]:


dataset["PPS_48"].value_counts()


# In[959]:


dataset["PPS_48"].isnull().sum()


# In[960]:


dataset["PPS_52"].value_counts()


# In[961]:


dataset["PPS_52"].isnull().sum()


# In[962]:


dataset["childage"].value_counts()


# In[963]:


dataset["childage"].isnull().sum()


# In[964]:


dataset.drop("childage", axis=1, inplace=True)


# In[965]:


dataset["cracorg1"].isnull().sum()


# In[966]:


dataset.drop("cracorg1", axis=1, inplace=True)


# In[967]:


dataset["_prace1"].isnull().sum()


# In[968]:


dataset.drop("_prace1", axis=1, inplace=True)


# In[969]:


dataset["mracasc1"].isnull().sum()


# In[970]:


dataset.drop("mracasc1", axis=1, inplace=True)


# In[971]:


dataset["_impcty"].isnull().sum()


# In[972]:


dataset.drop("_impcty", axis=1, inplace=True)


# In[973]:


dataset["mracorg1"].isnull().sum()


# In[974]:


dataset.drop("mracorg1", axis=1, inplace=True)


# In[975]:


dataset.shape


# In[976]:


# Exporting dataset:
from IPython.display import FileLink, FileLinks

dataset.to_csv('DataSet.csv', index=False)


# In[ ]:





# In[ ]:




