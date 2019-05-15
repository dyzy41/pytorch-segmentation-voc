# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:16:50 2018

@author: kawhi
"""
import sys  
import os, glob  
  
reload(sys)  
sys.setdefaultencoding('utf-8')  
  
#输出路径，自行修改  
TxtPath="train.txt"  
  
def BFS_Dir(dirPath, dirCallback = None, fileCallback = None):  
    queue = []  
    ret = []  
    f=open(TxtPath,'w')    # r只读，w可写，a追加  
    queue.append(dirPath);  
    while len(queue) > 0:  
        tmp = queue.pop(0)  
        if(os.path.isdir(tmp)):  
            ret.append(tmp)  
            for item in os.listdir(tmp):  
                queue.append(os.path.join(tmp, item))  
            if dirCallback:  
                dirCallback(tmp)  
        elif(os.path.isfile(tmp)):  
            ret.append(tmp)  
            if fileCallback:  
                mPath , ext = os.path.splitext(tmp)  
                names = os.path.split(mPath)  
                if(ext==".meta"):  
                    continue  
                else:  
                   print names[1]  
                   f.write(names[1])  
                   f.write('\n')  
                   fileCallback(tmp)  
    f.close()  
    return ret  
  
def printDir(dirPath):  
    print "dir: " + dirPath  
  
def printFile(dirPath):  
    print "file: " + dirPath  
  
if __name__ == '__main__':  
    while True:  
        path = "gt"
        
        try:  
            b = BFS_Dir(path , printDir, printFile)  
            print ("\r\n          *******\r\n"+"*********Done*********"+"\r\n          **********\r\n")  
        except:  
            print "Unexpected error:", sys.exc_info()  
        raw_input('press enter key to rehandle')
	exit()  
