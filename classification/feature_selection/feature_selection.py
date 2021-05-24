#!/usr/bin/env python
# coding: utf-8

# In[341]:


import pandas as pd
import numpy as np
import sys
sys.path.append("../..")
from data.dataset import DATASET as mainDf
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler


# In[316]:


mainDf.columns.values


# In[339]:


# write the necessary functions
# df = mainDf.loc[mainDf['risk1'].notnull()]
# t = np.where(np.isnan(x))

def getRowsForRisks(df, columns):
    partialDf = []
    for column in columns:
        data = df.loc[df[column].notnull()]
        partialDf.append(data)

    return partialDf

def preprocess(data, risks):
    data = data.drop(columns=['country','city','country_code','c40'])
    data = data.drop(columns=risks)

    return data

def getImportantFeatures(df, label, risks):
    y = df.loc[:, label]
    x = df.loc[:, df.columns != label]

    #risks.remove(label)
    #print(risks)
    
    x = x.drop(columns=['country','city','country_code','c40'])
    cols = [col for col in x.columns if col.lower()[:4] != 'risk']
    x = x[cols]

    #l = x.columns[x.isna().any()].tolist()
    #print(l)

    var_num = x.shape[1]
    var_num = int((var_num*15)/100)
    print("Picked variable number:",var_num)

    # Applying select K-best
    bestFeatures = SelectKBest(score_func=f_classif, k=var_num)
    fit = bestFeatures.fit(x,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)

    #print(dfscores)

    #Concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    print(featureScores.nlargest(var_num,'Score'))  #print 15% of the total features according to the score features

def getArrayOfFeatures(data, name):
    arr = data.columns.values
    return [s for s in arr if (name in s)]

def generateCrossFeat(data, arr):
    poly = PolynomialFeatures()
    crossed_feats = poly.fit_transform(data[arr].values)

    #Convert to Pandas DataFrame and merge to original dataset
    crossed_df = pd.DataFrame(crossed_feats)
    #print(crossed_df.shape)
    return crossed_df
    


# In[ ]:


#mainDf['Density'] = mainDf['']


# In[258]:


display(mainDf)
#print(mainDf.info)


# In[228]:


a = mainDf.columns.values.tolist()
#print(a)
risks = a[6:13]
risks_index = [i for i in range(6,14)]
print(risks)
print(risks_index)


# In[259]:


mylist = getRowsForRisks(mainDf, risks)
print(len(mylist))
for i in range(len(mylist)):
    print("Only",mylist[i].shape[0],"rows are available for risk",i)


# In[209]:


#display(mylist[0])


# In[230]:


for i in range(len(risks)):
    print("For the", risks[i])
    getImportantFeatures(mylist[i], risks[i], risks)


# In[ ]:


# Density features
# Try to get other algorithms


# # PCA Part - using standard scaler

# In[236]:


pca_df = preprocess(mainDf, risks)

val = pca_df.loc[:, pca_df.columns].values
print(val)

val = StandardScaler().fit_transform(val)
print(val.shape)


# In[237]:


np.mean(val), np.std(val)


# In[244]:


feat_cols = pca_df.columns.values.tolist()

normalized_val = pd.DataFrame(val, columns=feat_cols)
normalized_val.head()


# In[252]:


num_components = 10
pca_final = PCA(n_components=num_components)
pComponents_final = pca_final.fit_transform(val)
component_col = ['PC'+str(i+1) for i in range(num_components)]
print(component_col)

percentage_list = [element * 100 for element in pca_final.explained_variance_ratio_]
percentage_list = ['%.2f' % elem for elem in percentage_list]
print(percentage_list)


# In[253]:


pc_final_df = pd.DataFrame(data = pComponents_final, columns = component_col)
print(pc_final_df.shape)
pc_final_df.head()


# In[254]:


print('Explained variation percentage per principal component: {}'.format(percentage_list))
total_explained_percentage = (sum(pca_final.explained_variance_ratio_)*100)
print('Total percentage of the explained data by',pca_final.n_components,'components is: %.2f' %total_explained_percentage)
print('Percentage of the information that is lost for using',pca_final.n_components,'components is: %.2f' %(100-total_explained_percentage))


# In[255]:


mainDf_v2 = mainDf.merge(pc_final_df, left_index=True, right_index=True)
print(mainDf.shape)
print(mainDf_v2.shape)


# ## Doing the feature selection again after the addition of the Principal Components

# In[256]:


mylist2 = getRowsForRisks(mainDf_v2, risks)
print(len(mylist2))
for i in range(len(mylist2)):
    print("Only",mylist2[i].shape[0],"rows are available for risk",i)


# In[261]:


for i in range(len(risks)):
    print("-- For the", risks[i],"--")
    getImportantFeatures(mylist2[i], risks[i], risks)


# # PCA Part - using robust scaler

# In[263]:


pca_df = preprocess(mainDf, risks)

val = pca_df.loc[:, pca_df.columns].values
print(val)

val = RobustScaler().fit_transform(val)
print(val.shape)


# In[264]:


np.mean(val), np.std(val)


# In[265]:


feat_cols = pca_df.columns.values.tolist()

normalized_val = pd.DataFrame(val, columns=feat_cols)
normalized_val.head()


# In[266]:


num_components = 10
pca_final = PCA(n_components=num_components)
pComponents_final = pca_final.fit_transform(val)
component_col = ['PC'+str(i+1) for i in range(num_components)]
print(component_col)

percentage_list = [element * 100 for element in pca_final.explained_variance_ratio_]
percentage_list = ['%.2f' % elem for elem in percentage_list]
print(percentage_list)


# In[267]:


pc_final_df = pd.DataFrame(data = pComponents_final, columns = component_col)
print(pc_final_df.shape)
pc_final_df.head()


# In[268]:


print('Explained variation percentage per principal component: {}'.format(percentage_list))
total_explained_percentage = (sum(pca_final.explained_variance_ratio_)*100)
print('Total percentage of the explained data by',pca_final.n_components,'components is: %.2f' %total_explained_percentage)
print('Percentage of the information that is lost for using',pca_final.n_components,'components is: %.2f' %(100-total_explained_percentage))


# In[269]:


mainDf_v3 = mainDf.merge(pc_final_df, left_index=True, right_index=True)
print(mainDf.shape)
print(mainDf_v3.shape)


# ## Doing the feature selection again after the addition of the Principal Components

# In[270]:


mylist3 = getRowsForRisks(mainDf_v3, risks)
print(len(mylist3))
for i in range(len(mylist3)):
    print("Only",mylist3[i].shape[0],"rows are available for risk",i)


# In[271]:


for i in range(len(risks)):
    print("-- For the", risks[i],"--")
    getImportantFeatures(mylist3[i], risks[i], risks)


# # Try generating Polynomial Cross Features

# In[332]:


#other used words: population, index, %, female
pop_arr = getArrayOfFeatures(mainDf_v3, "population")
print(pop_arr)


# In[333]:


cross_df = generateCrossFeat(mainDf, pop_arr)
mainDf_v4 = mainDf.merge(cross_df, left_index=True, right_index=True)
print(mainDf.shape)
print(mainDf_v4.shape)


# In[342]:


print(mainDf_v4.shape)
mainDf_v4 = mainDf_v4.merge(pc_final_df, left_index=True, right_index=True)
print(mainDf_v4.shape)


# In[343]:


mainDf_v4.columns = mainDf_v4.columns.astype(str)


# In[344]:


mylist4 = getRowsForRisks(mainDf_v4, risks)
print(len(mylist4))
for i in range(len(mylist4)):
    print("Only",mylist4[i].shape[0],"rows are available for risk",i)


# In[345]:


for i in range(len(risks)):
    print("-- For the", risks[i],"--")
    getImportantFeatures(mylist4[i], risks[i], risks)


# In[ ]:




