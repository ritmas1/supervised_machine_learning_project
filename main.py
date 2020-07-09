#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:22:31 2020

@author: margaritavenediktova
"""

from classes import Classification
from itertools import product
import subprocess
from warnings import simplefilter

# ignore all future warnings       
simplefilter(action='ignore', category=FutureWarning)


	classifiers=['knn','rf']#,'ann'] # possible classifiers 

n_components_list=[2,3,4,5,6,9,10,15,30,70]
dim_red_list=['PCA','LDA','PCA+LDA']
smpl_list=['smote','undersampling','oversampling','unchanged']
combinations=list(product(smpl_list,dim_red_list,n_components_list))

#aux function to slice any number of classifiers
def pop_(list_,clf):
    for i in range(len(clf)):
        if len(list_)!=0:
            out=list_.pop(i)
            return out
        else:
            return 'classifier not specified'
        

with open('out.txt','w+') as out:
    with open('err.txt','w+') as err:
        for n in range(len(combinations[0:1])):
            clf=['rf','knn'] #chosen classifiers
            args = ['python','subprogram1.py',str(combinations[n][0]),str(combinations[n][1]),str(combinations[n][2]),pop_(clf,classifiers),pop_(clf,classifiers),pop_(clf,classifiers)]

            subprocess.run(args,stdout=out,stderr=err)
            out.seek(0)
            output=out.read()
    
            err.seek(0) 
            errors = err.read()
  
            
