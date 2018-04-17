#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import codecs

flag = False
is_ans = False
cat_name = "Computers&Internet"
fout = codecs.open(cat_name + ".txt", 'w', 'utf-8')
fans = codecs.open(cat_name + "_ans.txt", 'w', 'utf-8')
Max_item = 10000

with codecs.open('questions.txt', 'r', 'utf-8') as fin:
    i = 0
    for line in fin.readlines():
        if line[0:9] == "<maincat>":
            cat = line.strip().split(" ")[1]
            if cat == cat_name:
                flag = True
            else:
                if(flag):
                    break
        else:
            if flag and is_ans:
                fans.write(line)
                is_ans = False
            elif flag and not is_ans:
                fout.write(line)
                is_ans = True
                i += 1
                if i >= Max_item:
                    break

fout.close()

