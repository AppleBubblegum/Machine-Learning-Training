#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model


# In[2]:


#Preprocess raw data(replace every ',' with '.', for casting them into float)
f = open("k-means.csv", "r")
file_content = f.read()
file_content = file_content.replace(',', '.')
f = open("k-means_1.csv", "w")
f.write(file_content)
f.close()


# In[3]:


data = pd.read_csv("k-means_1.csv", sep = "\t", header = None)
data = data.rename(columns = { 0:'x', 1:'y'})
data["x"] = pd.to_numeric(data["x"], downcast="float")
data["y"] = pd.to_numeric(data["y"], downcast="float")
data_arr = data.to_numpy()
print(data)


# In[4]:


#Visualisation part
data.plot(x = 'x', y = 'y', kind = 'scatter')
plt.show()


# In[5]:


def K_means(data_arr):
    K=3
    m = data_arr.shape[0]
    n = data_arr.shape[1]
    iterat = 10
    import random
    # creating an empty centroid array
    centroids=np.array([]).reshape(n,0)   
    
    for k in range(K):
        centroids=np.c_[centroids,data_arr[random.randint(0,m-1)]]
        
    for i in range(iterat):
        euclid=np.array([]).reshape(m,0)
        for k in range(K):
            dist=np.sum((data_arr-centroids[:,k])**2,axis=1)
            euclid=np.c_[euclid,dist]
        C=np.argmin(euclid,axis=1)+1
        cent={}
        for k in range(K):
            cent[k+1]=np.array([]).reshape(2,0)
        for k in range(m):
            cent[C[k]]=np.c_[cent[C[k]],data_arr[k]]
        for k in range(K):
            cent[k+1]=cent[k+1].T
        for k in range(K):
            centroids[:,k]=np.mean(cent[k+1],axis=0)
        final=cent
    return final, centroids, K


# In[6]:


final, centroids, K = K_means(data.to_numpy())

for k in range(K):
    plt.scatter(final[k+1][:,0],final[k+1][:,1])
plt.scatter(centroids[0,:],centroids[1,:],s=300,c='yellow')
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
plt.show()


# In[7]:


#Testing part
test = np.array([0.3, 5.2, 2, 4.4, 1.1, 5.2, 5.2, 2.9, 5.3, 3.3, 6.1, 2.8])
test = test.reshape(6,2)
print(test)


# In[16]:


final, centroids, K = K_means(test)
for k in range(K):
    plt.scatter(final[k+1][:,0],final[k+1][:,1])
plt.scatter(centroids[0,:],centroids[1,:],s=100,c='yellow')
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
plt.show()


# In[9]:


from sklearn.cluster import KMeans
Cluster = KMeans(n_clusters=3, random_state = 2)
Cluster.fit(data)
y_pred = Cluster.predict(data)

plt.scatter(data_arr[:, 0], data_arr[:, 1], c=y_pred, s=50, cmap='plasma')
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})


# In[10]:


Cluster.fit(data)
y_pred = Cluster.predict(test)

plt.scatter(test[:, 0], test[:, 1], c=y_pred, s=50, cmap='plasma')
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})


# In[11]:


from sklearn.metrics import silhouette_score

for i in range(2, 10):
    clusterer = KMeans(n_clusters= i, random_state=i)
    cluster_labels = clusterer.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    print("For n_clusters =", i, "The average silhouette_score is :", silhouette_avg)


# In[ ]:




