#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import xml.sax
import codecs

# import sys
# print(sys.getdefaultencoding())

output_file = "questions.txt"

class MovieHandler( xml.sax.ContentHandler ):
    def __init__(self):
      self.CurrentData = ""
      self.subject = ""
      self.content = ""
      self.maincat = ""
      self.bestanswer = ""
      self.ques = {}
 
    # 元素开始事件处理 
    def startElement(self, tag, attributes):
        self.CurrentData = tag

 
   # 元素结束事件处理
    def endElement(self, tag):
        self.CurrentData = ""
        if tag == "maincat":
            question = self.subject + " " + self.content + "\n" + self.bestanswer
            if self.ques.has_key(self.maincat):
                self.ques[self.maincat].append(question)
            else:
                self.ques[self.maincat] = [question]
            self.maincat = ""
            self.subject = ""
            self.content = ""
            self.bestanswer = ""
        if tag == "ystfeed":
            with codecs.open(output_file, 'w', 'utf-8') as fout:
                for maincat, ques in self.ques.items():
                    print("maincat: ", maincat, "total questions: ", len(ques))
                    fout.write("<maincat> " + maincat + "\n")
                    for q in ques:
                        fout.write(q + "\n")
            #print(self.ques)
    
    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "subject":
            self.subject += content.strip()
        elif self.CurrentData == "content":
            self.content += content.strip()
        elif self.CurrentData == "maincat":
            self.maincat += content.strip()
        elif self.CurrentData == "bestanswer":
            self.bestanswer += content.strip()
            
    
if (  __name__ == "__main__"):
        
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    
    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler( Handler )
    
    parser.parse("FullOct2007.xml")
