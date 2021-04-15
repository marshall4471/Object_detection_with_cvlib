#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import cv2


# In[3]:


import cvlib as cv


# In[4]:


from cvlib.object_detection import draw_bbox


# In[5]:


im = cv2.imread('C://Users/Gaby/Desktop/beach_in_thailand.jpeg')


# In[6]:


bbbox, label, conf = cv.detect_common_objects(im)


# In[15]:


output_image = draw_bbox(im, bbbox, label, conf)


# In[18]:


plt.imshow(output_image)
plt.show()
cmap='gray'


# In[ ]:




