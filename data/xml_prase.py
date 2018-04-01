#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import xml.sax
# import sys
# print(sys.getdefaultencoding())

output_file = "questions.txt"

class MovieHandler( xml.sax.ContentHandler ):
    def __init__(self):
      self.CurrentData = ""
      self.subject = ""
      self.content = ""
      self.ques = ""
 
    # 元素开始事件处理 
    def startElement(self, tag, attributes):
        self.CurrentData = tag

 
   # 元素结束事件处理
    def endElement(self, tag):
        self.CurrentData = ""
        if tag == "content":
            self.ques = self.ques + self.subject + " " + self.content + "\n"
            self.subject = ""
            self.content = ""
        if tag == "ystfeed":
            with open(output_file, 'w') as fout:
                fout.write(self.ques)
            #print(self.ques)
    
    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "subject":
            self.subject += content.strip()
        elif self.CurrentData == "content":
            self.content += content.strip()
            
    
if (  __name__ == "__main__"):
        
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    
    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler( Handler )
    
    parser.parse("small_sample.xml")