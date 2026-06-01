#!/usr/bin/env python
# coding: utf-8

# <!-- Exploratory Data Analysis (EDA) on a Public Dataset -->

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_excel("titanic.xlsx")
df


# In[2]:


df.head(10)  #for knowing the strating elements


# In[3]:


df.tail(10)   #for knowing the ending elements


# In[4]:


df.sample(10)  #rondom row show


# In[5]:


df.iloc[50:65]  # show data in range


# In[6]:


df.info()  #show null values


# In[7]:


df.describe().round()


# In[8]:


df.isnull().sum()
# df.isnull().sum()/len(df)*10/100


# In[9]:


df.shape


# In[10]:


df.columns


# In[11]:


df.drop('deck',axis=1,inplace=True)


# In[52]:


df


# In[53]:


df["age"].median()


# In[54]:


df["age"].fillna(28,inplace=True)  #fill the age value



# In[55]:


df  #print all values


# In[56]:


df["age"]=df["age"].astype('int64')  #change float into int type


# In[18]:


df[df["embark_town"].isnull()]  # check null rows


# In[19]:


df["embark_town"].mode()  #mode of  embark_town


# In[20]:


df.drop('embarked',axis=1,inplace=True) # drop the colums


# In[58]:


df["embark_town"].fillna("Southampton",inplace=True)  # fill the value 


# In[59]:


df.isnull().sum()  # handle null values


# In[60]:


df.nunique()


# In[50]:


df[df['age']>=18]


# In[63]:


sns.countplot(x="alive",data=df,color="red")  # kitne log servive(1) vs not servive(0)
plt.title("alive count")
plt.show()


# In[64]:


#gender vs survived
sns.countplot(x="sex",hue="alive",data=df)
plt.title("gender vs alive")
plt.show()


# In[65]:


# plclass to survived
sns.countplot(x="pclass",hue="alive", data=df)
plt.title("class vs alive")
plt.show()


# In[66]:


#Age distribution

sns.histplot(df["age"],bins=30,kde=True)
plt.title("age distribution")  #Majority passengers young adults the(25-30)
plt.show()


# In[67]:


# df=df[df['fare']<500]
df["fare"].max()
# df["fare"].min()


# In[87]:


#fare vs survived
sns.boxplot(x="alive",y="fare",data=df)
plt.title("fare vs alive")
plt.show()


# In[30]:


df["fare"].describe()


# In[95]:


sns.histplot(df['fare'], kde=True) #fare count 
plt.title("fare distribution")
plt.show()


# In[ ]:


Q1 = df['fare'].quantile(0.25) #upper
Q1


# In[ ]:


Q2 = df['fare'].quantile(0.50) #median
Q2


# In[ ]:


Q3 = df['fare'].quantile(0.75) #lower
Q3


# In[40]:


IQR=Q3-Q1
IQR


# In[41]:


lower=Q1-1.5*IQR
lower


# In[43]:


upper=Q3+1.5*IQR
upper


# In[ ]:


df[df['fare']>upper]  #identify outliers

# df[df['fare']<lower]


# In[ ]:


df.drop("survived", axis=1,inplace=True)


# In[86]:


sns.boxplot(df['fare'])
plt.show()


# In[81]:


df.shape


# In[ ]:


import numpy as np
df["fare"]=np.clip(df["fare"],lower,upper)  # handle putlier fron capping mathod


# In[85]:


df.describe().round()


# In[ ]:


#correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation")
plt.show()


# In[94]:


df[(df["fare"]>=5) & (df["fare"]<=10)]


# In[100]:


df.isnull().sum()


# In[102]:


#save file
df.to_csv('clean_titanic.csv',index= False)
print("file sucessfully save !!")


# In[ ]:




