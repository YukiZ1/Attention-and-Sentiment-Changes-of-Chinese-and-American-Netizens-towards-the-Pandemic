#-*- coding : utf-8-*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jieba
import re
import time
from collections import Counter

#data=pd.read_csv(io,encoding='unicode_escape')
#path = 'E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//COVID-19-sentiment-analysis-dataset-Weibo-master//COVID-19 Unlabeled Weibo Data//'
path = 'E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//ll'
#path = 'E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//补充微博数据'
#path = 'E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//补充微博数据0102_0119'
files = os.listdir(path)#获取path路径下的所有子目录
'''
三组数据
1. 补充微博数据: 2020 0321 -- 2020 0331, 2020 0408--2020 0415
2. ll: 4.1--4.7
3. COVID-19 Unlabeled Weibo Data: 1.19-3.20
'''
#f = open('E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//COVID-19-sentiment-analysis-dataset-Weibo-master//COVID-19 Unlabeled Weibo Data//2.19.csv')

### Topic withdaw, test file:1.20.csv
idata = pd.read_csv('E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//COVID-19-sentiment-analysis-dataset-Weibo-master//COVID-19 Unlabeled Weibo Data//1.20.csv',encoding='utf-8')
#读取中文的编码有utf-8,gbk,gb18030,bg230,...共5种,encoding='unicode_escape'
#idata = pd.read_csv('E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//ll//4.1.csv',encoding='utf-8')#读取中文的编码有utf-8,gbk,gb18030,bg230,...共5种
print(idata.head())
topicdic = []
for index, row in idata.iterrows():
    #print(row['text'])
    itext = re.sub(r"\?|\【|\】",",",str(row['text']))#去除 【】
    #print(type(itext))
    itext = re.sub(r"@[^,，：:\s@]+", ",", itext)#去除 用户名
    itext = re.sub(r"#[^#]+#", ",",itext)#去除 #主题#
    itext = re.sub(r"展开全文c", "",itext)
    itext = re.sub(r"收起全文d", "",itext)
    itext = re.sub(r"O网页链接", "",itext)
    itext = re.sub(r"\s*", "",itext)#去除 空格
    #itext = itext.replace(" ","")
    #print(type(itext))
    #itext = re.findall(r"#.+#",itext,re.I|re.M)#re.I匹配对大小写不敏感，re.M多行匹配，影响^和$
    #print("\n\n===============\n"+itext)
    topicdic.append(itext)#添加itext
    #print("=============== \n",topicdic)
print("=============== 1.20 \n",topicdic)

def extract_message(idata):
    messagedic = []
    for index, row in idata.iterrows():
        #print(row['text'])
        itext = re.sub(r"@[^,，：:\s@]+", "", str(row['text']))#去除 用户名
        '''下面的是 <补充微博数据> 专享匹配服务'''
        #itext = re.sub(r"//@[^,，：:\s@]+", "", str(row['text']))#itext = re.sub(r"@[^,，：:\s@]+", "", str(row['text']))#去除 用户名
        itext = re.sub(r"\?|【|】", "", itext)
        itext = re.sub(r"#[^#]+#", "",itext)
        itext = re.sub(r"展开全文c", "",itext)
        itext = itext.replace('\u200b','')
        itext = re.sub(r"\s*", "",itext)#去除 空格
        #print(type(itext))
        #itext = re.findall(r"#.+#",itext,re.I|re.M)#re.I匹配对大小写不敏感，re.M多行匹配，影响^和$
        #print("=============== ",itext,"=========\n")
        messagedic.append(itext)#添加itext
    return messagedic

def extract_message2(idata):
    messagedic2 = []
    for index, row in idata.iterrows():
        #print(row['text'])
        itext = re.sub(r"@[^,，：:\s@]+", "", str(row['text']))#去除 用户名
        '''下面的是 <补充微博数据> 专享匹配服务'''
        #itext = re.sub(r"//@[^,，：:\s@]+", "", str(row['text']))#itext = re.sub(r"@[^,，：:\s@]+", "", str(row['text']))#去除 用户名
        itext = re.sub(r"【|】", "", itext)
        itext = re.sub(r"#", "",itext)
        itext = itext.replace('\u200b','')
        itext = re.sub(r"展开全文c", "",itext)
        itext = re.sub(r"收起全文d", "",itext)
        itext = re.sub(r"O网页链接", "",itext)
        itext = re.sub(r"\s*", "",itext)#去除 空格
        #print(type(itext))
        #itext = re.findall(r"#.+#",itext,re.I|re.M)#re.I匹配对大小写不敏感，re.M多行匹配，影响^和$
        #print("=============== ",itext,"=========\n")
        messagedic2.append(itext)#添加itext
    #print("\n\nextract_message2:",len(messagedic2),"\n",messagedic2[:3])
    return messagedic2

#topic_dic_total=[]#存储 每天topic的量, 可用来分析topic的变化
for ifile in files:
    if not os.path.isdir(ifile):#判断是否是文件夹，不是文件夹才打开
        print("\n============= idata ==========",ifile)
        idata = pd.read_csv(path+"//"+ifile,encoding='utf-8')#gb18030  utf-8
        
        #print("\n============= idata ==========",path,"//",ifile,"\n\n",idata)
        
        message_dic = {"original_text":idata['text'],"cleaned_text" : extract_message(idata),"removeWellSign":extract_message2(idata)}
        #topic_dic_total.append(topic_dic)#
        df = pd.DataFrame.from_dict(message_dic)
        #print(df)
        '''第一个是<补充微博数据> 专享文件命名方式
           第二个是<COVID-19 Unlabeled Weibo Data>和<ll>专享文件命名方式'''
        #df.to_csv("E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//微博数据处理//cleaned_text2//"+ifile[5]+'.'+str(int(ifile[6:8]))+".csv",encoding='utf-8',index=False)
        df.to_csv("E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//微博数据处理//cleaned_text2//"+ifile,encoding='utf-8',index=False)
        #print("\n*********",ifile,"\n\n",topic_dic)
