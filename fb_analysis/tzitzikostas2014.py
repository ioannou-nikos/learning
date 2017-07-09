# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:56:42 2017

This file is used to download data from tzitzikostas2014 facebook page. Remember to renew the token everytime.
@author: nikos
"""

import requests
import time
import pickle
import random


# Define the access token of facebook application
token = "EAACEdEose0cBAHOZC8n08tXYIKY58CiokZAzVRy2GdFNvXERx4H6ZBWpVA2NlcqRBci67snZAbJOy8oh88ySE6gNQt8m6cWqiEQEV8rVeYVq8HZCoT3AAcooZCwVpiPvu365CkF2zG6CuCBy3CANuiVzpqZATnJSuyLIl8ZBeVx6eF0wDJWp4DZCjAwQZAMpuDOoEZD"

def req_facebook(req):
    res = requests.get("https://graph.facebook.com/v2.8/" + req, {'access_token': token})
    return res

#req = "62782042702/posts?fields=comments.limit(0).summary(true),likes.limit(0).summary(true),message,created_time,link,object_id,updated_time,shares"
req = "131626260551551/posts?fields=comments.limit(0).summary(true),likes.limit(0).summary(true),message,created_time,link,object_id,updated_time,shares"
#req = "tzitzi2014?fields=posts{comments.limit(0).summary(true),likes.limit(0).summary(true),message}"
results = req_facebook(req).json()

# Define the array to hold the results
data = []

i = 0
while True:
    try:
        time.sleep(random.randint(2, 5))
        data.extend(results['data'])
        r = requests.get(results['paging']['next'])
        results = r.json()
        i += 1
        print(i)
        #if i > 2:
        #    break
    except:
        print("done")
        break

pickle.dump(data, open("kiriakosmitsotakis.pkl", "wb"))
