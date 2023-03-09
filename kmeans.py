#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math
from copy import deepcopy


# In[3]:


df = pd.read_csv('assignment2_input.txt', sep = '\t', names=['0','1','2','3','4','5','6','7','8','9','10','11'])
df.values


# In[3]:


k = 10


# In[4]:


def distance(x, y):
  distance = 0
  for i in range(len(x)):
    distance = (distance + (x[i] - y[i]) ** 2) ** 0.5
  return distance


# In[5]:


centroids = df.sample(k, random_state=50)


# In[6]:


labels = np.zeros(len(df))  # 각 데이터 포인트를 그룹화 할 labels을 생성

# 각 데이터를 순회하면서 centroids와의 거리를 측정
for i in range(len(df)):
  distances = np.zeros(k)
  for j in range(k):
    distances[j] = distance(df.values[i], centroids.values[j])
  cluster = np.argmin(distances)
  labels[i] = cluster


# In[7]:


centroids_old = deepcopy(centroids)


# In[8]:


centroids_old = np.zeros(centroids.shape) # 제일 처음 centroids_old는 0으로 초기화
labels = np.zeros(len(df))

error = np.zeros(k) # error 초기화
for i in range(k):
  error[i] = distance(centroids_old[i], centroids.values[i])

# STEP 4: error가 0에 수렴할 때 까지 2 ~ 3 단계를 반복
while(error.all() != 0):                   
    # STEP 2: 가까운 centroids에 데이터를 할당
  for i in range(len(df)):
    distances = np.zeros(k)	       # 초기 거리는 모두 0으로 초기화
    for j in range(k):
      distances[j] = distance(df.values[i], centroids.values[j])
    cluster = np.argmin(distances)
    labels[i] = cluster
    
    # Step 3: centroids를 업데이트 시켜줌
  centroids_old = deepcopy(centroids.values)
  for i in range(k):
    # 각 그룹에 속한 데이터들만 골라 points에 저장
    points = [df.values[j] for j in range(len(df)) if labels[j] == i]
    
    # points의 각 feature, 즉 각 좌표의 평균 지점을 centroid로 지정
    centroids.values[i] = np.mean(points, axis=0)
    
  # 새롭게 centroids를 업데이트 했으니 error를 다시 계산  
  for i in range(k):
    error[i] = distance(centroids_old[i], centroids.values[i])


# In[9]:


df


# In[10]:


cluster = pd.DataFrame(labels)


# In[11]:


df['cluster_num'] = cluster


# In[12]:


df


# In[26]:


for i in range (0,10):
    print(len(df.loc[df['cluster_num'] == i]),":",df.loc[df['cluster_num'] == i].index.tolist())







