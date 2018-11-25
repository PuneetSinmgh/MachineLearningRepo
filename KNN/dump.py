# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:56:00 2018

@author: PuneetPC
"""

import pickle
import numpy as np

bush=[0.154638053,0.64935962]    

williams=[0.2063057929,0.5372413793]
    
pickle.dump((bush), open('bush.pkl', 'wb'))

pickle.dump((williams), open('williams.pkl', 'wb'))

print("DONE")
