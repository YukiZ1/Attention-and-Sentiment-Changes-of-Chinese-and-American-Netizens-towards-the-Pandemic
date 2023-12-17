import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path='E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//00 微博结果分析//weibo_final_svm//'
files = os.listdir(path)

counts_twitter=[]#每天的推特条数
timeseries_emotion=[]#0、1、2占比
timeseries_topic=[]#0-9每个话题占比

timeseries_emotion0=[]#-1、1情绪占比
timeseries_emotion1=[]
timeseries_topic0=[]#0-4每个话题占比
timeseries_topic1=[]
timeseries_topic2=[]
timeseries_topic3=[]
timeseries_topic4=[]

topic_emotion00=[]#每种topic的情绪#topic0,情绪0
topic_emotion01=[]#topic0,情绪1
topic_emotion10=[]
topic_emotion11=[]
topic_emotion20=[]
topic_emotion21=[]
topic_emotion30=[]
topic_emotion31=[]
topic_emotion40=[]
topic_emotion41=[]

'''
# 1. 单个测试
idata = pd.read_csv(path + '2020-2-03.csv',encoding='utf-8')
print(idata.head())
ilen=len(idata)
counts_twitter.append(ilen)

emotion0=len(idata.loc[idata['label3']==0,:])/ilen
emotion1=len(idata.loc[idata['label3']==1,:])
emotion2=len(idata.loc[idata['label3']==2,:])

tp0=len(idata.loc[idata['Dominant_Topic']==0.0,:])/ilen
tp1=len(idata.loc[idata['Dominant_Topic']==1.0,:])
tp2=len(idata.loc[idata['Dominant_Topic']==2.0,:])
tp3=len(idata.loc[idata['Dominant_Topic']==3.0,:])
tp4=len(idata.loc[idata['Dominant_Topic']==4.0,:])
tp5=len(idata.loc[idata['Dominant_Topic']==5.0,:])
tp6=len(idata.loc[idata['Dominant_Topic']==6.0,:])
tp7=len(idata.loc[idata['Dominant_Topic']==7.0,:])
tp8=len(idata.loc[idata['Dominant_Topic']==8.0,:])
tp9=len(idata.loc[idata['Dominant_Topic']==9.0,:])
#tp9=len(idata.loc[:,:][idata['Dominant_Topic']==9.0])
'''
'''
for index.row in idata.itterows():
    emotion1 = idata.[][idata]
'''
ctp=[0,0,0,0,0]

ilen=0
# 2. 数据处理
for ifile in files:
    if not os.path.isdir(ifile):
        idata = pd.read_csv(path + ifile,encoding='utf-8')
        ilen=len(idata)
        counts_twitter.append(ilen)
        #row['Dominant_Topic']
        #row['label3']
        #每种情感的时间变化
        timeseries_emotion0.append(len(idata.loc[idata['label']==-1,:])/ilen)#每种情绪数/推特总数
        timeseries_emotion1.append(len(idata.loc[idata['label']==1,:])/ilen)
                
        idata0=idata.loc[idata['Dominant_Topic']==0.0,:]
        idata1=idata.loc[idata['Dominant_Topic']==1.0,:]
        idata2=idata.loc[idata['Dominant_Topic']==2.0,:]
        idata3=idata.loc[idata['Dominant_Topic']==3.0,:]
        idata4=idata.loc[idata['Dominant_Topic']==4.0,:]
        
        ilen0=len(idata0)+1#每种话题的条数
        ilen1=len(idata1)+1
        ilen2=len(idata2)+1
        ilen3=len(idata3)+1
        ilen4=len(idata4)+1
        
        ctp[0]+=ilen0#每种话题的总数
        ctp[1]+=ilen1
        ctp[2]+=ilen2
        ctp[3]+=ilen3
        ctp[4]+=ilen4
        
        #每种topic占比的时间变化
        timeseries_topic0.append(ilen0/ilen)#每种话题数/推特总数
        timeseries_topic1.append(ilen1/ilen)
        timeseries_topic2.append(ilen2/ilen)
        timeseries_topic3.append(ilen3/ilen)
        timeseries_topic4.append(ilen4/ilen)
        
        #每种topic的情感变化
        topic_emotion00.append(len(idata0.loc[idata0['label']==-1,:])/ilen0)#该话题消极数/该话题数总数
        topic_emotion01.append(len(idata0.loc[idata0['label']==1,:])/ilen0)#该话题积极数/该话题数总数
        topic_emotion10.append(len(idata1.loc[idata1['label']==-1,:])/ilen1)
        topic_emotion11.append(len(idata1.loc[idata1['label']==1,:])/ilen1)
        topic_emotion20.append(len(idata2.loc[idata2['label']==-1,:])/ilen2)
        topic_emotion21.append(len(idata2.loc[idata2['label']==1,:])/ilen2)
        topic_emotion30.append(len(idata3.loc[idata3['label']==-1,:])/ilen3)
        topic_emotion31.append(len(idata3.loc[idata3['label']==1,:])/ilen3)
        topic_emotion40.append(len(idata4.loc[idata4['label']==-1,:])/ilen4)
        topic_emotion41.append(len(idata4.loc[idata4['label']==1,:])/ilen4)
        
#x=pd.date_range('1/2/2020','1/2/2020')
#x2=pd.date_range('1/8/2020','1/9/2020')
#x3=pd.date_range('1/11/2020','1/11/2020')
#x4=pd.date_range('1/14/2020','1/15/2020')
x5=pd.date_range('1/19/2020','3/2/2020')
x6=pd.date_range('3/10/2020','4/15/2020')
#x=x.append([x2,x3,x4,x5,x6])
x=x5.append(x6)
message_dic = {"date":x,"消极占比" : timeseries_emotion0,"积极占比" : timeseries_emotion1,"r_topic0" : timeseries_topic0,"r_topic1" : timeseries_topic1,"r_topic2" : timeseries_topic2,"r_topic3" : timeseries_topic3,"r_topic4" : timeseries_topic4}
df = pd.DataFrame.from_dict(message_dic)
df.to_csv("E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//00 微博结果分析//hot//resultAll.csv",encoding='utf-8',index=False)
'''
tpe0=[topic_emotion01[i]+topic_emotion00[i]*(-1) for i in range(len(topic_emotion00))]
tpe1=[topic_emotion11[i]+topic_emotion10[i]*(-1) for i in range(len(topic_emotion10))]
tpe2=[topic_emotion21[i]+topic_emotion20[i]*(-1) for i in range(len(topic_emotion20))]
tpe3=[topic_emotion31[i]+topic_emotion30[i]*(-1) for i in range(len(topic_emotion30))]
tpe4=[topic_emotion41[i]+topic_emotion40[i]*(-1) for i in range(len(topic_emotion40))]
message_dic2 = {"date":x,"s_topic0" : tpe0,"s_topic1" : tpe1,"s_topic2" : tpe2,"s_topic3" : tpe3,"s_topic4" : tpe4}
df2 = pd.DataFrame.from_dict(message_dic2)
df2.to_csv("E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//00 微博结果分析//hot//weibo_tpe.csv",encoding='utf-8',index=False)
'''
# 3. 画图
import matplotlib
matplotlib.rc("font",family='YouYuan')
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
plt.rcParams['figure.figsize'] = (12.0, 8.0)
#plt.rcParams.update({"font.size":20})
'''
plt.plot(x,tpe0,label=u'话题0 全球疫情发展',color='navajowhite')
plt.plot(x,tpe1,label=u'话题1 专家呼吁告诫防护',color='mediumpurple')
plt.plot(x,tpe2,label=u'话题2 国内疫情确诊状况',color='thistle')
plt.plot(x,tpe3,label=u'话题3 各级防疫政策及个人防疫情况',color='tan')
plt.plot(x,tpe4,label=u'话题4 正能量致敬医护团队',color='darkseagreen')

plt.rcParams.update({'font.size':7})
plt.legend(loc='lower right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('微博话题情绪得分变化图', fontproperties=font_set, fontsize=20)
plt.savefig('./微博话题情绪得分变化图')
plt.show()
'''
'''
plt.plot(x,counts_twitter,c='lightseagreen')
plt.grid()
plt.title('微博数目变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./微博数目变化图')
plt.show()
'''
#plt.plot(x,timeseries_emotion0,c='turquoise',label=u'消极')
#plt.plot(x,timeseries_emotion1,c='bisque',label=u'积极')
plt.plot(x,timeseries_emotion0,label=u'消极',color='mediumpurple')
plt.plot(x,timeseries_emotion1,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('宏观情感占比变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./宏观情感占比变化图')
plt.show()

labels=[u'话题0 祈福武汉致敬医护人员',u'话题1 感慨与愿景',u'话题2 ',u'话题3 全球疫情发展',u'话题4 国内疫情形势']
colors = ['tan','navajowhite','lightsalmon','coral','tomato','lightcoral','powderblue','mediumturquoise','lightseagreen','cadetblue']
plt.pie(ctp,labels=labels,autopct='%1.2f%%',colors=colors)
plt.title('各话题总体占比饼图', fontproperties=font_set,fontsize=20)
plt.savefig('./宏各话题总体占比饼图')
plt.show()

plt.plot(x,timeseries_topic0,label=u'话题0 祈福武汉致敬医护人员',color='blueviolet')
plt.plot(x,timeseries_topic1,label=u'话题1 感慨与愿景',color='thistle')
plt.plot(x,timeseries_topic2,label=u'话题2 各级防疫政策及个人防疫情况',color='darkseagreen')
plt.plot(x,timeseries_topic3,label=u'话题3 全球疫情发展',color='orange')
plt.plot(x,timeseries_topic4,label=u'话题4 国内疫情形势',color='darkturquoise')
#plt.plot(x,timeseries_topic5,label=u'话题5 防护治疗',color='beige')
#plt.plot(x,timeseries_topic6,label=u'话题6 对领导政府的评价',color='darkturquoise')
#plt.plot(x,timeseries_topic7,label=u'话题7 国际疫情状况',color='darkseagreen')
#plt.plot(x,timeseries_topic8,label=u'话题8 美国检测确诊情况',color='wheat')
#plt.plot(x,timeseries_topic9,label=u'话题9 情绪表达',color='powderblue')
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('宏观话题占比变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./宏观话题占比变化图')
plt.show()

plt.plot(x,topic_emotion00,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion01,label=u'积极',color='mediumturquoise')
#plt.axvline(x='4/10/2020',ls="-",c="red")#添加垂直直线
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('话题0"祈福武汉致敬医护人员"情绪变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./话题0情绪变化图')
plt.show()

plt.plot(x,topic_emotion10,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion11,label=u'积极',color='mediumturquoise')
#plt.vlines(76, 0, 0.5, colors = "red", linestyles = "dashed")
#plt.axvline(x='4/10/2020',ls="-",c="red")#添加垂直直线
plt.legend(loc='upper right')
plt.grid()
plt.title('话题1"感慨与愿景"情绪变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./话题1情绪变化图')
plt.show()

plt.plot(x,topic_emotion20,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion21,label=u'积极',color='mediumturquoise')
#plt.axvline(x='4/10/2020',ls="-",c="red")#添加垂直直线
plt.legend(loc='upper right')
plt.grid()
plt.title('话题2"各级防疫政策及个人防疫情况"情绪变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./话题2情绪变化图')
plt.show()

plt.plot(x,topic_emotion30,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion31,label=u'积极',color='mediumturquoise')
#plt.axvline(x=23,ls="-",c="red")#添加垂直直线
plt.legend(loc='upper right')
plt.grid()
plt.title('话题3"全球疫情发展"情绪变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./话题3情绪变化图')
plt.show()

plt.plot(x,topic_emotion40,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion41,label=u'积极',color='mediumturquoise')
#plt.axvline(x='4/10/2020',ls="-",c="red")#添加垂直直线
plt.legend(loc='upper right')
plt.grid()
plt.title('话题4"国内疫情形势"情绪变化图', fontproperties=font_set,fontsize=20)
plt.savefig('./话题4情绪变化图')
plt.show()

'''
plt.plot(x,topic_emotion50,label=u'消极')
plt.plot(x,topic_emotion51,label=u'中立')
plt.plot(x,topic_emotion52,label=u'积极')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题5"防护治疗"情绪变化图', fontproperties=font_set)
plt.savefig('./话题5情绪变化图')
plt.show()

plt.plot(x,topic_emotion60,label=u'消极')
plt.plot(x,topic_emotion61,label=u'中立')
plt.plot(x,topic_emotion62,label=u'积极')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题6"对领导政府的评价"情绪变化图', fontproperties=font_set)
plt.savefig('./话题6情绪变化图')
plt.show()

plt.plot(x,topic_emotion70,label=u'消极')
plt.plot(x,topic_emotion71,label=u'中立')
plt.plot(x,topic_emotion72,label=u'积极')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题7"国际疫情状况"情绪变化图', fontproperties=font_set)
plt.savefig('./话题7情绪变化图')
plt.show()

plt.plot(x,topic_emotion80,label=u'消极')
plt.plot(x,topic_emotion81,label=u'中立')
plt.plot(x,topic_emotion82,label=u'积极')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题8"美国检测确诊情况"情绪变化图', fontproperties=font_set)
plt.savefig('./话题8情绪变化图')
plt.show()
            
plt.plot(x,topic_emotion90,label=u'消极')
plt.plot(x,topic_emotion91,label=u'中立')
plt.plot(x,topic_emotion92,label=u'积极')
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()
plt.title('话题9"情绪表达"情绪变化图', fontproperties=font_set)
plt.savefig('./话题9情绪变化图')
plt.show()
'''
