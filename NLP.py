#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import nltk
from bs4 import BeautifulSoup
response=urllib.request.urlopen('http://php.net/')
html=response.read()
raw=BeautifulSoup(html,"html5lib").get_text();

