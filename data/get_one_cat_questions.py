#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import codecs

flag = False
cat_name = "Computers&Internet"
fout = codecs.open(cat_name + ".txt", 'w', 'utf-8')

with codecs.open('questions.txt', 'r', 'utf-8') as fin:
    for line in fin.readlines():
        if line[0:9] == "<maincat>":
            cat = line.strip().split(" ")[1]
            if cat == cat_name:
                flag = True
            else:
                if(flag):
                    break
        else:
            if(flag):
                fout.write(line)

fout.close()

