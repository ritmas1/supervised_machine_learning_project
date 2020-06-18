#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:29:08 2020

@author: margaritavenediktova
"""

from classes import Classification
import sys        


example=Classification(sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5],sys.argv[6])

example.split()
example.classification_process()
