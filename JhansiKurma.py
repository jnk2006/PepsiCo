#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install the following packages in command prompt
'''pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install plotly==3.10
pip install sklearn
pip install more-itertools
'''


# # Pepsi Challenge
# 
# ### Reading data

# In[2]:


import numpy as np
import pandas as pd

df = pd.read_excel('C:/Users/Naveena/Documents/pepsi challenge/shelf-life-study-data-for-analytics-challenge-v2.xlsx')

df.head(10)


# In[3]:


for i in df.columns.values:
    print(i)
    print(df[i].value_counts())
    print('--------------------')


# <p>Numerical variables are:
#     <ul>
#         <li>Study Number</li>
#         <li>Sample ID</li>
#         <li>Sample Age</li>
#         <li>Difference From Fresh</li>
#         <li>Processing Agent Stability Index</li>
#         <li>Moisture</li>
#         <li>Residual oxygen</li>
#         <li>Hexanal</li>
#     </ul></p>
# <p>Categorical variables are:
#     <ul>
#         <li>Product Type</li>
#         <li>Base Ingredient</li>
#         <li>Process Type</li>
#         <li>Storage Conditions</li>
#         <li>Packaging stabilizer added</li>
#         <li>Transparent window in package</li>
#         <li>Preservative Added</li>
#     </ul></p>

# In[4]:


#Checking the datatypes
df.dtypes


# In[5]:


#Checking the numerical data
df.describe()


# In[6]:


#Check the categorical variables
df.describe(include = 'object')


# In[7]:


#Providing concise summary of dataframe
df.info()


# ### Data Wrangling

# #### 1. Handling Missing Values

# In[8]:


#Let's see how many missing values there are in the dataframe
print(df.isnull().sum())


# <p>The missing value variables include 
#     <ul>Categorical
#         <li>Base Ingredient</li>
#         <li>Storage Conditions</li>
#         <li>Packaging Stabilizer Added</li>
#         <li>Transparent Window in Package</li>
#         <li>Preservative Added</li>
#     </ul>
#     <ul>Numerical
#     <li>Moisture (%)</li>
#     <li>Residual Oxygen (%)</li>
#     <li>Hexanal (ppm)</li>
#     </ul>
# </p>

# <p>Since the owner of the dataset mentioned that the information of the missing data was either not measured or captured, we can replace the NaN values with "not mentioned/captured".</p>

# In[9]:


df.replace(np.nan, 'Not Mentioned/Captured', inplace = True)


# In[10]:


df.head()


# #### 2. Data Binning
# 
# <p>If 'Difference From Fresh' > 20.0, the product is said to be no longer fresh. Hence, let's change the values of 'Difference From Fresh' greater than 20 to 0 and remaining values to 1.</p>

# In[11]:


#Store the Difference From Fresh column in a different dataframe to perform binning
df_dff = df[['Difference From Fresh']]
df_dff['Difference From Fresh'] = df_dff['Difference From Fresh'].astype('int')

def classify(row):
    if ((row['Difference From Fresh'] >= 0) and (row['Difference From Fresh'] <= 20)):
        return '1'
    else:
        return '0'
        
df_dff['Difference From Fresh Labels'] = df_dff.apply(lambda row: classify(row), axis=1)
df_dff['Difference From Fresh Labels'] = df_dff['Difference From Fresh Labels'].astype('int')
df_dff.head(10)


# In[12]:


df_dff.drop('Difference From Fresh', axis=1, inplace = True)
df_dff.head()


# In[13]:


#Concatenate the dataframes
df = pd.concat([df, df_dff], axis = 1)
df.head(10)


# In[14]:


#Let's check how many of the 749 products are fresh and not fresh.
df['Difference From Fresh Labels'].value_counts()


# There are 612 cases that aren't fresh and 137 that are fresh.

# #### 3. Dummy variables
# 
# <p>Let's create dummy variables for the categorical variables of the dataframe that consists of only the important features.</p>
# <p>Let's take only those columns that are important for us.</p>
# <ul>
#     <li>Product type</li>
#     <li>Base Ingredient</li>
#     <li>Process Type</li>
#     <li>Storage Conditions</li>
#     <li>Packaging Stabilizer Added</li>
#     <li>Transparent Window in Package</li>
#     <li>Processing Agent Stabililty Index</li>
#     <li>Preservative Added</li>

# In[15]:


f_df = df[['Product Type', 'Base Ingredient', 'Process Type', 'Storage Conditions', 'Packaging Stabilizer Added', 'Transparent Window in Package', 'Processing Agent Stability Index', 'Preservative Added', 'Difference From Fresh', 'Difference From Fresh Labels']]
print(f_df.shape)
f_df.head(10)


# In[16]:


f_df_dummy = pd.get_dummies(data=f_df, columns=f_df.select_dtypes(include = ['object']).columns)
f_df_dummy.head()


# In[17]:


f_df_dummy.shape


# In[18]:


f_df_dummy.columns.tolist()


# <p>We can now see that the number of columns increased from 10 to 34 due to the one-hot encoding. Moreover, f_df_dummy dataframe will be used later on while performing <b>normalization</b>. And f_df dataframe will be used from now on as it contains important features for determining shelf life of a product.</p>

# ### Exploratory Data Analysis
# 
# <p>Let's perform some visualizations to understand data better.</p>

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

import plotly as plty
import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in(username='jnk22', api_key='Ds3oWpnL2ULkxlVddHRo')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


f_df.head()


# #### 1. Product Type vs Difference from Fresh and Difference From Fresh Labels

# In[21]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (20,20))

f_df[f_df['Product Type'] == 'A'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[0, 0])
axes[0,0].set_xlabel("Product Type A")
axes[0,0].set_ylabel('Differnce From Fresh')
axes[0,0].get_legend().remove()
axes[0,0].set_xticks([], [])

f_df[f_df['Product Type'] == 'B'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[0,1])
axes[0,1].set_xlabel("Product Type B")
axes[0,1].set_ylabel('Differnce From Fresh')
axes[0,1].get_legend().remove()
axes[0,1].set_xticks([], [])

f_df[f_df['Product Type'] == 'C'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[0,2])
axes[0,2].set_xlabel("Product Type C")
axes[0,2].set_ylabel('Differnce From Fresh')
axes[0,2].get_legend().remove()
axes[0,2].set_xticks([], [])

f_df[f_df['Product Type'] == 'D'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[1,0])
axes[1,0].set_xlabel("Product Type D")
axes[1,0].set_ylabel('Differnce From Fresh')
axes[1,0].get_legend().remove()
axes[1,0].set_xticks([], [])

f_df[f_df['Product Type'] == 'E'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[1, 1])
axes[1,1].set_xlabel("Product Type E")
axes[1,1].set_ylabel('Differnce From Fresh')
axes[1,1].get_legend().remove()
axes[1,1].set_xticks([], [])

f_df[f_df['Product Type'] == 'F'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[1, 2])
axes[1,2].set_xlabel("Product Type F")
axes[1,2].set_ylabel('Differnce From Fresh')
axes[1,2].get_legend().remove()
axes[1,2].set_xticks([], [])

f_df[f_df['Product Type'] == 'G'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[2, 0])
axes[2,0].set_xlabel("Product Type G")
axes[2,0].set_ylabel('Differnce From Fresh')
axes[2,0].get_legend().remove()
axes[2,0].set_xticks([], [])

f_df[f_df['Product Type'] == 'H'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[2, 1])
axes[2,1].set_xlabel("Product Type H")
axes[2,1].set_ylabel('Differnce From Fresh')
axes[2,1].get_legend().remove()
axes[2,1].set_xticks([], [])

f_df[f_df['Product Type'] == 'I'].plot(x = 'Product Type', y = 'Difference From Fresh', kind = 'line', ax = axes[2, 2])
axes[2,2].set_xlabel("Product Type I")
axes[2,2].set_ylabel('Differnce From Fresh')
axes[2,2].get_legend().remove()
axes[2,2].set_xticks([], [])

plt.show()


# In[22]:


grouped = f_df.groupby(['Product Type', 'Difference From Fresh Labels'])['Product Type'].count()
grouped = grouped.to_frame()
grouped.rename(columns = {'Product Type': 'Count'}, inplace = True)
grouped.index.names = ['Product Type', 'FreshLabels']
grouped.reset_index(inplace = True)
grouped = grouped.set_index('FreshLabels')

x = grouped['Product Type'].unique()
sl1 = []
sl0 = []
for index, row in grouped.iterrows():
    if(index == 0):
        sl0.append(row[1])
    else:
        sl1.append(row[1])

shelflife1 = go.Bar(x = x, y = sl1, text = sl1, textposition = 'auto', name = 'Unexpired (Shelf Life < 20)')
shelflife0 = go.Bar(x = x, y = sl0, text = sl0, textposition = 'auto', name = 'Expired (Shelf Life > 20)')
data = [shelflife1, shelflife0]
layout = go.Layout(barmode = 'group', title = 'Product Type and Shelf Life')

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='Product Type and Shelf Life')


# #### 2. Base Ingredient vs Difference from Fresh and Difference From Fresh Labels

# In[23]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (20,10))

f_df[f_df['Base Ingredient'] == 'A'].plot(x = 'Base Ingredient', y = 'Difference From Fresh', kind = 'line', ax = axes[0, 0])
axes[0,0].set_xlabel("Base Ingredient A")
axes[0,0].set_ylabel('Difference From Fresh')
axes[0,0].get_legend().remove()
axes[0,0].set_xticks([], [])

f_df[f_df['Base Ingredient'] == 'B'].plot(x = 'Base Ingredient', y = 'Difference From Fresh', kind = 'line', ax = axes[0,1])
axes[0,1].set_xlabel("Base Ingredient B")
axes[0,1].set_ylabel('Differnce From Fresh')
axes[0,1].get_legend().remove()
axes[0,1].set_xticks([], [])

f_df[f_df['Base Ingredient'] == 'C'].plot(x = 'Base Ingredient', y = 'Difference From Fresh', kind = 'line', ax = axes[0,2])
axes[0,2].set_xlabel("Base Ingredient C")
axes[0,2].set_ylabel('Difference From Fresh')
axes[0,2].get_legend().remove()
axes[0,2].set_xticks([], [])

f_df[f_df['Base Ingredient'] == 'D'].plot(x = 'Base Ingredient', y = 'Difference From Fresh', kind = 'line', ax = axes[1,0])
axes[1,0].set_xlabel("Base Ingredient D")
axes[1,0].set_ylabel('Difference From Fresh')
axes[1,0].get_legend().remove()
axes[1,0].set_xticks([], [])

f_df[f_df['Base Ingredient'] == 'E'].plot(x = 'Base Ingredient', y = 'Difference From Fresh', kind = 'line', ax = axes[1, 1])
axes[1,1].set_xlabel("Base Ingredient E")
axes[1,1].set_ylabel('Difference From Fresh')
axes[1,1].get_legend().remove()
axes[1,1].set_xticks([], [])

f_df[f_df['Base Ingredient'] == 'F'].plot(x = 'Base Ingredient', y = 'Difference From Fresh', kind = 'line', ax = axes[1, 2])
axes[1,2].set_xlabel("Base Ingredient F")
axes[1,2].set_ylabel('Difference From Fresh')
axes[1,2].get_legend().remove()
axes[1,2].set_xticks([], [])

fig1, ax1 = plt.subplots(figsize = (6,6))
f_df[f_df['Base Ingredient'] == 'Not Mentioned/Captured'].plot(x = 'Base Ingredient', y = 'Difference From Fresh', kind = 'line', ax = ax1)
ax1.set_xlabel("Base Ingredient Not Mentioned/Captured")
ax1.set_ylabel('Difference From Fresh')
ax1.get_legend().remove()
ax1.set_xticks([], [])

plt.show()


# In[24]:


grouped = f_df.groupby(['Base Ingredient', 'Difference From Fresh Labels'])['Base Ingredient'].count()
grouped = grouped.to_frame()
grouped.rename(columns = {'Base Ingredient': 'Count'}, inplace = True)
grouped.index.names = ['Base Ingredient', 'FreshLabels']
grouped.reset_index(inplace = True)
grouped = grouped.set_index('FreshLabels')

x = grouped['Base Ingredient'].unique()
sl1 = []
sl0 = []
for index, row in grouped.iterrows():
    if(index == 0):
        sl0.append(row[1])
    else:
        sl1.append(row[1])

shelflife1 = go.Bar(x = x, y = sl1, text = sl1, textposition = 'auto', name = 'Unexpired (Shelf Life < 20)')
shelflife0 = go.Bar(x = x, y = sl0, text = sl0, textposition = 'auto', name = 'Expired (Shelf Life > 20)')
data = [shelflife1, shelflife0]
layout = go.Layout(barmode = 'group', title = 'Base Ingredient and Shelf Life')

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='Base Ingredient and Shelf Life')


# #### 3. Process Type vs Difference from Fresh and Difference From Fresh Labels

# In[25]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

f_df[f_df['Process Type'] == 'A'].plot(x = 'Process Type', y = 'Difference From Fresh', kind = 'line', ax = axes[0])
axes[0].set_xlabel("Process Type A")
axes[0].set_ylabel('Difference From Fresh')
axes[0].get_legend().remove()
axes[0].set_xticks([], [])

f_df[f_df['Process Type'] == 'B'].plot(x = 'Process Type', y = 'Difference From Fresh', kind = 'line', ax = axes[1])
axes[1].set_xlabel("Process Type B")
axes[1].set_ylabel('Differnce From Fresh')
axes[1].get_legend().remove()
axes[1].set_xticks([], [])

fig1, ax1 = plt.subplots(figsize = (15,6))
f_df[f_df['Process Type'] == 'C'].plot(x = 'Process Type', y = 'Difference From Fresh', kind = 'line', ax = ax1)
ax1.set_xlabel("Process Type C")
ax1.set_ylabel('Difference From Fresh')
ax1.get_legend().remove()
ax1.set_xticks([], [])

plt.show()


# In[26]:


grouped = f_df.groupby(['Process Type', 'Difference From Fresh Labels'])['Process Type'].count()
grouped = grouped.to_frame()
grouped.rename(columns = {'Process Type': 'Count'}, inplace = True)
grouped.index.names = ['Process Type', 'FreshLabels']
grouped.reset_index(inplace = True)
grouped = grouped.set_index('FreshLabels')

x = grouped['Process Type'].unique()
sl1 = []
sl0 = []
for index, row in grouped.iterrows():
    if(index == 0):
        sl0.append(row[1])
    else:
        sl1.append(row[1])

shelflife1 = go.Bar(x = x, y = sl1, text = sl1, textposition = 'auto', name = 'Unexpired (Shelf Life < 20)')
shelflife0 = go.Bar(x = x, y = sl0, text = sl0, textposition = 'auto', name = 'Expired (Shelf Life > 20)')
data = [shelflife1, shelflife0]
layout = go.Layout(barmode = 'group', title = 'Process Type and Shelf Life')

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='Process Type and Shelf Life')


# #### 4. Storage Conditions vs Difference from Fresh and Difference From Fresh Labels

# In[27]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (20,10))

f_df[f_df['Storage Conditions'] == 'Cold Climate'].plot(x = 'Storage Conditions', y = 'Difference From Fresh', kind = 'line', ax = axes[0, 0])
axes[0,0].set_xlabel("Storage Conditions - Cold Climate")
axes[0,0].set_ylabel('Difference From Fresh')
axes[0,0].get_legend().remove()
axes[0,0].set_xticks([], [])

f_df[f_df['Storage Conditions'] == 'High Temperature and Humidity'].plot(x = 'Storage Conditions', y = 'Difference From Fresh', kind = 'line', ax = axes[0,1])
axes[0,1].set_xlabel("Storage Conditions - High Temperature and Humidity")
axes[0,1].set_ylabel('Differnce From Fresh')
axes[0,1].get_legend().remove()
axes[0,1].set_xticks([], [])

f_df[f_df['Storage Conditions'] == 'Not Mentioned/Captured'].plot(x = 'Storage Conditions', y = 'Difference From Fresh', kind = 'line', ax = axes[1,0])
axes[1,0].set_xlabel("Storage Conditions - Not Mentioned/Captured")
axes[1,0].set_ylabel('Difference From Fresh')
axes[1,0].get_legend().remove()
axes[1,0].set_xticks([], [])

f_df[f_df['Storage Conditions'] == 'Warm Climate'].plot(x = 'Storage Conditions', y = 'Difference From Fresh', kind = 'line', ax = axes[1, 1])
axes[1,1].set_xlabel("Storage Conditions - Warm Climate")
axes[1,1].set_ylabel('Difference From Fresh')
axes[1,1].get_legend().remove()
axes[1,1].set_xticks([], [])

plt.show()


# In[28]:


grouped = f_df.groupby(['Storage Conditions', 'Difference From Fresh Labels'])['Storage Conditions'].count()
grouped = grouped.to_frame()
grouped.rename(columns = {'Storage Conditions': 'Count'}, inplace = True)
grouped.index.names = ['Storage Conditions', 'FreshLabels']
grouped.reset_index(inplace = True)
grouped = grouped.set_index('FreshLabels')

x = grouped['Storage Conditions'].unique()
sl1 = []
sl0 = []
for index, row in grouped.iterrows():
    if(index == 0):
        sl0.append(row[1])
    else:
        sl1.append(row[1])

shelflife1 = go.Bar(x = x, y = sl1, text = sl1, textposition = 'auto', name = 'Unexpired (Shelf Life < 20)')
shelflife0 = go.Bar(x = x, y = sl0, text = sl0, textposition = 'auto', name = 'Expired (Shelf Life > 20)')
data = [shelflife1, shelflife0]
layout = go.Layout(barmode = 'group', title = 'Storage Conditions and Shelf Life')

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='Storage Conditions and Shelf Life')


# #### 5. Packaging Stabilizer Added vs Difference from Fresh and Difference From Fresh Labels

# In[29]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

f_df[f_df['Packaging Stabilizer Added'] == 'N'].plot(x = 'Packaging Stabilizer Added', y = 'Difference From Fresh', kind = 'line', ax = axes[0])
axes[0].set_xlabel("Packaging Stabilizer Added - N")
axes[0].set_ylabel('Difference From Fresh')
axes[0].get_legend().remove()
axes[0].set_xticks([], [])

f_df[f_df['Packaging Stabilizer Added'] == 'Y'].plot(x = 'Packaging Stabilizer Added', y = 'Difference From Fresh', kind = 'line', ax = axes[1])
axes[1].set_xlabel("Packaging Stabilizer Added - Y")
axes[1].set_ylabel('Differnce From Fresh')
axes[1].get_legend().remove()
axes[1].set_xticks([], [])

fig1, ax1 = plt.subplots(figsize = (15,6))
f_df[f_df['Packaging Stabilizer Added'] == 'Not Mentioned/Captured'].plot(x = 'Packaging Stabilizer Added', y = 'Difference From Fresh', kind = 'line', ax = ax1)
ax1.set_xlabel("Packaging Stabilizer Added - Not Mentioned/Captured")
ax1.set_ylabel('Difference From Fresh')
ax1.get_legend().remove()
ax1.set_xticks([], [])

plt.show()


# In[30]:


grouped = f_df.groupby(['Packaging Stabilizer Added', 'Difference From Fresh Labels'])['Packaging Stabilizer Added'].count()
grouped = grouped.to_frame()
grouped.rename(columns = {'Packaging Stabilizer Added': 'Count'}, inplace = True)
grouped.index.names = ['Packaging Stabilizer Added', 'FreshLabels']
grouped.reset_index(inplace = True)
grouped = grouped.set_index('FreshLabels')

x = grouped['Packaging Stabilizer Added'].unique()
sl1 = []
sl0 = []
for index, row in grouped.iterrows():
    if(index == 0):
        sl0.append(row[1])
    else:
        sl1.append(row[1])

shelflife1 = go.Bar(x = x, y = sl1, text = sl1, textposition = 'auto', name = 'Unexpired (Shelf Life < 20)')
shelflife0 = go.Bar(x = x, y = sl0, text = sl0, textposition = 'auto', name = 'Expired (Shelf Life > 20)')
data = [shelflife1, shelflife0]
layout = go.Layout(barmode = 'group', title = 'Packaging Stabilizer Added and Shelf Life')

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='Packaging Stabilizer Added and Shelf Life')


# #### 6. Transparent Window in Package vs Difference from Fresh and Difference From Fresh Labels

# In[31]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

f_df[f_df['Transparent Window in Package'] == 'N'].plot(x = 'Transparent Window in Package', y = 'Difference From Fresh', kind = 'line', ax = axes[0])
axes[0].set_xlabel("Transparent Window in Package - N")
axes[0].set_ylabel('Difference From Fresh')
axes[0].get_legend().remove()
axes[0].set_xticks([], [])

f_df[f_df['Transparent Window in Package'] == 'Not Mentioned/Captured'].plot(x = 'Transparent Window in Package', y = 'Difference From Fresh', kind = 'line', ax = axes[1])
axes[1].set_xlabel("Transparent Window in Package - Not Mentioned/Captured")
axes[1].set_ylabel('Differnce From Fresh')
axes[1].get_legend().remove()
axes[1].set_xticks([], [])

plt.show()


# In[32]:


grouped = f_df.groupby(['Transparent Window in Package', 'Difference From Fresh Labels'])['Transparent Window in Package'].count()
grouped = grouped.to_frame()
grouped.rename(columns = {'Transparent Window in Package': 'Count'}, inplace = True)
grouped.index.names = ['Transparent Window in Package', 'FreshLabels']
grouped.reset_index(inplace = True)
grouped = grouped.set_index('FreshLabels')

x = grouped['Transparent Window in Package'].unique()
sl1 = []
sl0 = []
for index, row in grouped.iterrows():
    if(index == 0):
        sl0.append(row[1])
    else:
        sl1.append(row[1])

shelflife1 = go.Bar(x = x, y = sl1, text = sl1, textposition = 'auto', name = 'Unexpired (Shelf Life < 20)')
shelflife0 = go.Bar(x = x, y = sl0, text = sl0, textposition = 'auto', name = 'Expired (Shelf Life > 20)')
data = [shelflife1, shelflife0]
layout = go.Layout(barmode = 'group', title = 'Transparent Window in Package and Shelf Life')

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='Transparent Window in Package and Shelf Life')


# #### 6. Preservative Added vs Difference from Fresh and Difference From Fresh Labels

# In[33]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

f_df[f_df['Preservative Added'] == 'N'].plot(x = 'Preservative Added', y = 'Difference From Fresh', kind = 'line', ax = axes[0])
axes[0].set_xlabel("Preservative Added - N")
axes[0].set_ylabel('Difference From Fresh')
axes[0].get_legend().remove()
axes[0].set_xticks([], [])

f_df[f_df['Preservative Added'] == 'Y'].plot(x = 'Preservative Added', y = 'Difference From Fresh', kind = 'line', ax = axes[1])
axes[1].set_xlabel("Preservative Added - Y")
axes[1].set_ylabel('Differnce From Fresh')
axes[1].get_legend().remove()
axes[1].set_xticks([], [])

fig1, ax1 = plt.subplots(figsize = (15,6))
f_df[f_df['Preservative Added'] == 'Not Mentioned/Captured'].plot(x = 'Preservative Added', y = 'Difference From Fresh', kind = 'line', ax = ax1)
ax1.set_xlabel("Preservative Added - Not Mentioned/Captured")
ax1.set_ylabel('Difference From Fresh')
ax1.get_legend().remove()
ax1.set_xticks([], [])

plt.show()


# In[34]:


grouped = f_df.groupby(['Preservative Added', 'Difference From Fresh Labels'])['Preservative Added'].count()
grouped = grouped.to_frame()
grouped.rename(columns = {'Preservative Added': 'Count'}, inplace = True)
grouped.index.names = ['Preservative Added', 'FreshLabels']
grouped.reset_index(inplace = True)
grouped = grouped.set_index('FreshLabels')

x = grouped['Preservative Added'].unique()
sl1 = []
sl0 = []
for index, row in grouped.iterrows():
    if(index == 0):
        sl0.append(row[1])
    else:
        sl1.append(row[1])

shelflife1 = go.Bar(x = x, y = sl1, text = sl1, textposition = 'auto', name = 'Unexpired (Shelf Life < 20)')
shelflife0 = go.Bar(x = x, y = sl0, text = sl0, textposition = 'auto', name = 'Expired (Shelf Life > 20)')
data = [shelflife1, shelflife0]
layout = go.Layout(barmode = 'group', title = 'Preservative Added and Shelf Life')

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='Preservative Added and Shelf Life')


# #### 7. Processing Agent Stability Index vs Difference from Fresh and Difference From Fresh Labels

# In[35]:


fig, ax = plt.subplots(figsize = (10, 10))
plt.title('Regplot of Processing agent stability index vs Difference from fresh')
sns.regplot(x = 'Processing Agent Stability Index', y = 'Difference From Fresh', data = f_df, ax = ax)
plt.ylim(0,)


# #### 8. Correlation Heatmap

# In[36]:


fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(f_df.corr(), ax = ax, annot = True, linewidths = 0.3, cmap = 'RdYlGn')
plt.title('Shelf Life Dataset Correlation Heatmap', fontsize = 18)
plt.show()
#fig.savefig('heatmap_of_all_features.png')


# ### Data Pre-processing and selection

# In[37]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# <p>From here on, f_df_dummy dataframe will be used for normalization and data modeling.</p>

# In[38]:


print(f_df_dummy.shape)
f_df_dummy.head()


# #### 1. Splitting data

# In[39]:


x0 = f_df_dummy.iloc[:, 0]
x = f_df_dummy.iloc[:,3:]
x = pd.concat([x, x0], axis = 1)
x = np.asarray(x)
x[0:5]


# In[40]:


y = f_df_dummy.iloc[:,2:3]
y = np.asarray(y)
y[0:5]


# #### 2. Normalizing data

# In[41]:


x = StandardScaler().fit(x).transform(x)
x[0:5]


# ### Train/Test Data

# In[42]:


#split the data in 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# ### Modeling - Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[44]:


#Trainig the dataset
lr = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
lr


# In[45]:


#Making predictions using the test set
yhat_lr = lr.predict(x_test)
yhat_lr


# <b>predict_proba</b> returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):

# In[46]:


yhat_prob_lr = lr.predict_proba(x_test)
yhat_prob_lr


# ### Evaluation using Jaccard index and Log loss functions

# #### 1. Jaccard Index
# 
# <p>Lets try jaccard index for accuracy evaluation. We can define jaccard as the size of the intersection divided by the size of the union of two label sets. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.</p>

# In[47]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat_lr)


# #### 2. Log loss
# 
# <p>Now, lets try <b>log loss</b> for evaluation. In logistic regression, the output can be the probability of product expired (equal to 0; 'Difference From Fresh' > 20). This probability is a value between 0 and 1.</p>
# <p>Log loss (Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.</p>

# In[48]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob_lr)


# #### Confusion Matrix

# In[49]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect = 'auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat_lr, labels=[1,0]))


# In[50]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_lr, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure(figsize = (8,8))
plot_confusion_matrix(cnf_matrix, classes=['Difference From Fresh <= 20','Difference From Fresh > 20'],normalize= False,  title='Logistic Regression Confusion matrix')


# In[51]:


print (classification_report(y_test, yhat_lr))


# ### Modeling - Logistic Regression Cross Validation

# In[52]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold


# In[53]:


#Create a function to perform logistic regression using cross validation for various cv values.
def logisticregressioncv(num):
    kf = KFold(n_splits=num)
    clf = LogisticRegressionCV(cv=kf, random_state=0, multi_class='auto', solver = 'liblinear').fit(x, y)
    yhatclf = clf.predict(x)
    yhatclf_proba = clf.predict_proba(x)
    score = clf.score(x, y)
    print('Number of folds: {}, Score: {}'.format(num, score))

#Pass 2 to 15 folds for to the function
num = np.asarray(list(range(2,16,1)))
for i in num:
    logisticregressioncv(i)


# <p> We can see that all the folds have the score of 0.8397863818424566.</p>

# In[54]:


clf = LogisticRegressionCV(cv=10, random_state=0, multi_class='auto', solver = 'liblinear').fit(x, y)
yhat_lrcv = clf.predict(x)
yhat_proba_lrcv = clf.predict_proba(x)
score_lrcv = clf.score(x, y)
print('Number of folds: {}, Score: {}'.format(clf.cv, score_lrcv))


# In[55]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y, yhat_lrcv, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure(figsize = (8,8))
plot_confusion_matrix(cnf_matrix, classes=['Difference From Fresh <= 20','Difference From Fresh > 20'],normalize= False,  title='Logistic Regression CV Confusion matrix')


# In[56]:


print (classification_report(y, yhat_lrcv))


# ### Modeling - SVM

# In[57]:


from sklearn import svm
clf_svm = svm.SVC(kernel='rbf', probability = True)
clf_svm.fit(x_train, y_train)


# In[58]:


yhat_svm = clf_svm.predict(x_test)
yhat_svm[0:5]


# In[59]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_svm, labels=[1,0])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_svm))

# Plot non-normalized confusion matrix
plt.figure(figsize = (8,8))
plot_confusion_matrix(cnf_matrix, classes=['Difference From Fresh <= 20','Difference From Fresh > 20'],normalize= False,  title='SVM Confusion matrix')


# In[60]:


yhat_proba_svm = clf_svm.predict_proba(x_test)
yhat_proba_svm


# In[61]:


jaccard_similarity_score(y_test, yhat_svm)


# ### Comparing Algorithms

# #### 1. Bar chart - Accuracy score

# In[62]:


from sklearn.metrics import accuracy_score

y_axis = np.array([accuracy_score(y_test, yhat_lr), accuracy_score(y, yhat_lrcv), accuracy_score(y_test, yhat_svm)])
x_axis = ['Logistic Regression', 'Logistic Regression Cross Validatioin', 'SVM' ]
fig, ax = plt.subplots(figsize = (12, 10))
ax1 = plt.bar(x_axis, y_axis)
plt.title("Comparison of Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")

def add_value_labels(ax1, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.5f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# Call the function above. All the magic happens there.
add_value_labels(ax1)
plt.show()


# #### 2. ROC / AUC curve

# In[63]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

lr_roc_auc = roc_auc_score(y_test, yhat_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, yhat_prob_lr[:,1])
axes[0].plot(fpr_lr, tpr_lr, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
axes[0].plot([0, 1], [0, 1],'r--')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc="lower right")

svm_roc_auc = roc_auc_score(y_test, yhat_svm)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, yhat_proba_svm[:,1])
axes[1].plot(fpr_svm, tpr_svm, label='SVM (area = %0.2f)' % svm_roc_auc)
axes[1].plot([0, 1], [0, 1],'r--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc="lower right")

fig1, ax1 = plt.subplots(figsize = (15,6))
lrcv_roc_auc = roc_auc_score(y, yhat_lrcv)
fpr_lrcv, tpr_lrcv, thresholds_lrcv = roc_curve(y, yhat_proba_lrcv[:,1])
ax1.plot(fpr_lrcv, tpr_lrcv, label='Logistic Regression CV (area = %0.2f)' % lrcv_roc_auc)
ax1.plot([0, 1], [0, 1],'r--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc="lower right")


plt.title('Receiver operating characteristic')
plt.show()


# <p>Since LogisticRegressionCV() has better ROC/AUC curve and high accuracy compared to Logistic Regression and SVM, results of the former mentioned model will be concatenated to the initial dataframe.</p>

# ### Adding results to the dataframe

# In[64]:


print(yhat_lrcv.shape)
yhat_lrcv


# In[65]:


print(yhat_proba_lrcv.shape)
yhat_proba_lrcv


# In[66]:


dataset = pd.DataFrame({'Numeric result': yhat_lrcv})
dataset.head(10)


# In[67]:


def description(row):
    if row['Numeric result'] == 1:
        return 'Fresh'
    else:
        return 'No longer fresh'


# In[68]:


dataset['Result Description'] = dataset.apply (lambda row: description(row), axis=1)


# In[69]:


dataset.head(10)


# In[70]:


dataset['Numeric result'].value_counts()


# In[71]:


dataset = pd.concat([dataset, df], axis = 1)
dataset.head(10)


# In[72]:


dataset.drop('Difference From Fresh Labels', axis = 1, inplace = True)
dataset.head()


# ### Add the final dataframe to an excel sheet

# In[73]:


dataset.to_excel (r'C:\Users\Naveena\Desktop\JhansiKurma.xlsx', index = None, header=True, sheet_name = 'Classification Description') #Don't forget to add '.xlsx' at the end of the path


# In[ ]:




