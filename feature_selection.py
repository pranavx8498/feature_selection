#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv('mobile.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


x=df.iloc[:,0:-1]
x.head()
y=df.iloc[:,-1:]
y.head()


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=5)


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


lr=LinearRegression()


# In[9]:


lr.fit(x_train,y_train)


# In[10]:


lr.predict(x_test)


# In[11]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[12]:


ranking=SelectKBest(score_func=chi2,k=20)


# In[13]:


ranking.fit(x,y)


# In[14]:


odf=ranking.scores_


# In[15]:


fd=pd.DataFrame(odf,columns=['Score'])
nfd=pd.DataFrame(x.columns,columns=['feature'])
feature_score=pd.concat([nfd,fd],axis=1)


# In[16]:


feature_score.nlargest(10,'Score')


# # feature importance

# In[17]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[18]:


met=ExtraTreesClassifier()


# In[19]:


met.fit(x,y)


# In[20]:


rank=pd.Series(met.feature_importances_,index=x.columns)


# In[21]:


rank.nlargest(10).plot(kind='bar')


# # Correlation

# In[22]:


import seaborn as sns
df.corr


# In[23]:


top=df.corr()


# In[24]:


top_index=top.index


# In[25]:


plt.figure(figsize=(20,20))
sns.heatmap(df[top_index].corr(),annot=True)


# In[26]:



# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[27]:


from sklearn.feature_selection import mutual_info_classif 


# In[28]:


mutual_info=mutual_info_classif(x,y)
mutual_data=pd.Series(mutual_info,index=x.columns)
mutual_data.sort_values(ascending=False)


# # variance thereshold 
# it will remove the valur which threshhold value maching

# In[29]:


from sklearn.feature_selection import VarianceThreshold
vt=VarianceThreshold(threshold=1)
vt.fit(x)


# In[30]:


x.columns[vt.get_support()]


# In[31]:


zero_therhold=[ i for i in x.columns
               if i not in x.columns[vt.get_support()]]


# In[34]:


df=pd.read_csv('train.csv')


# In[32]:


len(zero_therhold)


# In[33]:


for names in zero_therhold:
    print(names)


# In[44]:


x=df.drop('TARGET',axis=1)
y=df['TARGET']


# In[37]:


from sklearn.model_selection import train_test_split


# In[41]:


df.columns


# In[43]:


from sklearn.feature_selection import VarianceThreshold
vt=VarianceThreshold(threshold=0)


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=10)


# In[46]:


vt.fit(x_train)


# In[55]:


constant_columns = [column for column in x_train.columns
                    if column not in x_train.columns[vt.get_support()]]


# In[57]:


df.drop(constant_columns,axis=1,inplace=True)


# In[59]:


from sklearn.datasets import load_boston


# In[60]:


dd=load_boston()


# In[64]:


df=pd.DataFrame(dd.data,columns=dd.feature_names)


# In[68]:


df["MEDV"] = dd.target


# In[69]:



X = df.drop("MEDV",axis=1)
y = df["MEDV"]


# In[70]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0)


# In[ ]:




