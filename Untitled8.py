#!/usr/bin/env python
# coding: utf-8

# # Task 3: Predicting optimum number of clusters and representing it visually.

# ## Author: Nidhi Lohani

# We are using Kmeans clustering algorithm to get clusters. This is unsupervised algorithm. K defines the number of pre defined clusters that need to be created in the process. This is done by elbow method, which is based on concept of wcss (within cluster sum of squares).

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#To display maximum columns of dataframe
pd.pandas.set_option('display.max_columns',None)


# ## Loading dataset

# In[3]:


data=pd.read_csv('C:\\Users\\LOHANI\\Desktop\\Iris2.csv')
print("Data imported")


# In[4]:


print(data.shape)


# In[5]:


data.head()


# ## Extracting Independent variables

# In[6]:


X=data.iloc[:,[0,1,2,3]].values


# ## finding optimum value of k

# In[10]:


from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    Kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
    
#plotting the results into line graph
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("No of clusters")
plt.ylabel("WCSS")
plt.show()


# ## Using dendogram to find optimal no of clusters. 

# ## Hierarchical clustering

# In[12]:


import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title("Dendrogram")
plt.xlabel("Species")
plt.ylabel("Euclidean Distance")
plt.show()


# optimum clusters will be cluster after which wcss remains almost constant. From above two graphs, optimum no of clusters is 3.

# ## creating kmeans classifier

# In[13]:


kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)


# ## Visualizing the clusters

# In[15]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='setosa')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='versicolor')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='virginica')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='centroids')
plt.legend()


# In[ ]:




