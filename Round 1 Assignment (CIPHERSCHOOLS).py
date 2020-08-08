#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf


# In[52]:


df= pd.read_csv('C:/Users/user/Desktop/haberman.csv',header=None, names=['age', 'year_of_treatment', 'positive_lymph_nodes', 'survival_status_after_5_years'])


# In[53]:


df


# In[54]:


print(df.info())


# 1. No Null Value is present in the dataset.
# 2. We have to convert survival_status_after_5_years into categorical variables to perform the analysis.

# # Change into categorical variable:

# In[55]:


df['survival_status_after_5_years'] = df['survival_status_after_5_years'].map({1:"yes", 2:"no"})
df['survival_status_after_5_years'] = df['survival_status_after_5_years'].astype('category')


# In[56]:


print(df.head())


# In[ ]:





# # High Level Statistics

# In[57]:


df.describe()


# In[58]:


print("Number of rows: " + str(df.shape[0]))
print("Number of columns: " + str(df.shape[1]))
print("Columns: " + ", ".join(df.columns))

print("Target variable distribution")
print(df.iloc[:,-1].value_counts())
print("*"*50)
print(df.iloc[:,-1].value_counts(normalize = True))


# 1. Dataset is very small(i.e, 306) which effect the result, for predictive amchine learning size  of dataset must be large
# 2. No. of features :4 
# 3. No. of Classes : 2("YES" ,"NO")
# 4. Approx. 74% have the positive_lymph_nodes and 24% doesn't have it.
# 5. The dataset doesn't show the feature of a balanced dataset with 74% "YES".

# In[ ]:





# 
# # OBJECTIVE:
# 
# 
# To predict whether the patient will survive after 5 years or not based upon the patient's age, year of treatment and the number of positive lymph nodes

# In[ ]:





# In[21]:


df.hist(figsize=(14,14))
plt.show()


# In[ ]:





# In[26]:


px.bar(df,x='survival_status_after_5_years',y='positive_lymph_nodes')


# In[ ]:





# # PDF

# In[16]:


counts,bin_edges=np.histogram(df["positive_lymph_nodes"],bins=10,density=True)

pdf1=counts/sum(counts)


# In[18]:


print(pdf1)


# In[19]:


counts,bin_edges=np.histogram(df["year_of_treatment"],bins=10,density=True)

pdf2=counts/sum(counts)


# In[20]:


print(pdf2)


# In[ ]:





# 

# In[32]:


for idx, feature in enumerate(list(df.columns)[:-1]):
    fg = sns.FacetGrid(df, hue='survival_status_after_5_years', height=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()


# # Observation1:

# In[ ]:


# No. of positive lymph nodes is highly densed from 0 to 1.


# In[ ]:





# # BOXPLOT

# In[38]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(df.columns)[:-1]):
    sns.boxplot( x='survival_status_after_5_years', y=feature, data=df, ax=axes[idx])
plt.show() 


# In[ ]:





# # VIOLIN PLOT

# In[39]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(df.columns)[:-1]):
    sns.violinplot( x='survival_status_after_5_years', y=feature, data=df, ax=axes[idx])
plt.show()


# In[ ]:





# # Observation2:
#   

# In[ ]:


# The patient treated after 1966 have slightly more chances to survive  as compared to previous ones.


# In[ ]:





# # Bivariate Analysis:

# In[ ]:





# In[42]:


sns.pairplot(df, hue='survival_status_after_5_years', height=4)
plt.show()


# In[ ]:





# In[50]:


px.scatter(df,x='age',y='positive_lymph_nodes')


# In[49]:


px.scatter(df,x='year_of_treatment',y='positive_lymph_nodes')


# # Observation:

# In[ ]:


# Year of treatment and positve lymph nodes has better separation than any other sactter plot.
# There is a high positive lymph nodes for the people having age between 40-70 years.

