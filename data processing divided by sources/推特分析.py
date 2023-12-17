import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path='E://S.T.U.D.Y/S.T.U.D.Y【4.1】//数据挖掘//小组大任务//00 推特结果分析//tweet_final_2//'
files = os.listdir(path)

counts_twitter=[]#每天的推特条数
timeseries_emotion=[]#0、1、2占比
timeseries_topic=[]#0-9每个话题占比

timeseries_emotion0=[]#0、1、2情绪占比
timeseries_emotion1=[]
timeseries_emotion2=[]
timeseries_topic0=[]#0-9每个话题占比
timeseries_topic1=[]
timeseries_topic2=[]
timeseries_topic3=[]
timeseries_topic4=[]
timeseries_topic5=[]
timeseries_topic6=[]
timeseries_topic7=[]
timeseries_topic8=[]
timeseries_topic9=[]
topic_emotion00=[]#每种topic的情绪#topic0,情绪0
topic_emotion01=[]#topic0,情绪1
topic_emotion02=[]#topic0,情绪2
topic_emotion10=[]
topic_emotion11=[]
topic_emotion12=[]
topic_emotion20=[]
topic_emotion21=[]
topic_emotion22=[]
topic_emotion30=[]
topic_emotion31=[]
topic_emotion32=[]
topic_emotion40=[]
topic_emotion41=[]
topic_emotion42=[]
topic_emotion50=[]
topic_emotion51=[]
topic_emotion52=[]
topic_emotion60=[]
topic_emotion61=[]
topic_emotion62=[]
topic_emotion70=[]
topic_emotion71=[]
topic_emotion72=[]
topic_emotion80=[]
topic_emotion81=[]
topic_emotion82=[]
topic_emotion90=[]
topic_emotion91=[]
topic_emotion92=[]
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
ctp=[0,0,0,0,0,0,0,0,0,0]

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
        timeseries_emotion0.append(len(idata.loc[idata['label3']==0,:])/ilen)#每种情绪数/推特总数
        timeseries_emotion1.append(len(idata.loc[idata['label3']==1,:])/ilen)
        timeseries_emotion2.append(len(idata.loc[idata['label3']==2,:])/ilen)
        
        idata0=idata.loc[idata['Dominant_Topic']==0.0,:]
        idata1=idata.loc[idata['Dominant_Topic']==1.0,:]
        idata2=idata.loc[idata['Dominant_Topic']==2.0,:]
        idata3=idata.loc[idata['Dominant_Topic']==3.0,:]
        idata4=idata.loc[idata['Dominant_Topic']==4.0,:]
        idata5=idata.loc[idata['Dominant_Topic']==5.0,:]
        idata6=idata.loc[idata['Dominant_Topic']==6.0,:]
        idata7=idata.loc[idata['Dominant_Topic']==7.0,:]
        idata8=idata.loc[idata['Dominant_Topic']==8.0,:]
        idata9=idata.loc[idata['Dominant_Topic']==9.0,:]
        ilen0=len(idata0)#每种话题的条数
        ilen1=len(idata1)
        ilen2=len(idata2)
        ilen3=len(idata3)
        ilen4=len(idata4)
        ilen5=len(idata5)
        ilen6=len(idata6)
        ilen7=len(idata7)
        ilen8=len(idata8)
        ilen9=len(idata9)
        ctp[0]+=ilen0#每种话题的总数
        ctp[1]+=ilen1
        ctp[2]+=ilen2
        ctp[3]+=ilen3
        ctp[4]+=ilen4
        ctp[5]+=ilen5
        ctp[6]+=ilen6
        ctp[7]+=ilen7
        ctp[8]+=ilen8
        ctp[9]+=ilen9
        #每种topic占比的时间变化
        timeseries_topic0.append(ilen0/ilen)#每种话题数/推特总数
        timeseries_topic1.append(ilen1/ilen)
        timeseries_topic2.append(ilen2/ilen)
        timeseries_topic3.append(ilen3/ilen)
        timeseries_topic4.append(ilen4/ilen)
        timeseries_topic5.append(ilen5/ilen)
        timeseries_topic6.append(ilen6/ilen)
        timeseries_topic7.append(ilen7/ilen)
        timeseries_topic8.append(ilen8/ilen)
        timeseries_topic9.append(ilen9/ilen)
        #每种topic的情感变化
        topic_emotion00.append(len(idata0.loc[idata0['label3']==0,:])/ilen0)#该话题消极数/该话题数总数
        topic_emotion01.append(len(idata0.loc[idata0['label3']==1,:])/ilen0)#该话题中立数/该话题数总数
        topic_emotion02.append(len(idata0.loc[idata0['label3']==2,:])/ilen0)#该话题积极数/该话题数总数
        topic_emotion10.append(len(idata1.loc[idata1['label3']==0,:])/ilen1)
        topic_emotion11.append(len(idata1.loc[idata1['label3']==1,:])/ilen1)
        topic_emotion12.append(len(idata1.loc[idata1['label3']==2,:])/ilen1)
        topic_emotion20.append(len(idata2.loc[idata2['label3']==0,:])/ilen2)
        topic_emotion21.append(len(idata2.loc[idata2['label3']==1,:])/ilen2)
        topic_emotion22.append(len(idata2.loc[idata2['label3']==2,:])/ilen2)
        topic_emotion30.append(len(idata3.loc[idata3['label3']==0,:])/ilen3)
        topic_emotion31.append(len(idata3.loc[idata3['label3']==1,:])/ilen3)
        topic_emotion32.append(len(idata3.loc[idata3['label3']==2,:])/ilen3)
        topic_emotion40.append(len(idata4.loc[idata4['label3']==0,:])/ilen4)
        topic_emotion41.append(len(idata4.loc[idata4['label3']==1,:])/ilen4)
        topic_emotion42.append(len(idata4.loc[idata4['label3']==2,:])/ilen4)
        topic_emotion50.append(len(idata5.loc[idata5['label3']==0,:])/ilen5)
        topic_emotion51.append(len(idata5.loc[idata5['label3']==1,:])/ilen5)
        topic_emotion52.append(len(idata5.loc[idata5['label3']==2,:])/ilen5)
        topic_emotion60.append(len(idata6.loc[idata6['label3']==0,:])/ilen6)
        topic_emotion61.append(len(idata6.loc[idata6['label3']==1,:])/ilen6)
        topic_emotion62.append(len(idata6.loc[idata6['label3']==2,:])/ilen6)
        topic_emotion70.append(len(idata7.loc[idata7['label3']==0,:])/ilen7)
        topic_emotion71.append(len(idata7.loc[idata7['label3']==1,:])/ilen7)
        topic_emotion72.append(len(idata7.loc[idata7['label3']==2,:])/ilen7)
        topic_emotion80.append(len(idata8.loc[idata8['label3']==0,:])/ilen8)
        topic_emotion81.append(len(idata8.loc[idata8['label3']==1,:])/ilen8)
        topic_emotion82.append(len(idata8.loc[idata8['label3']==2,:])/ilen8)
        topic_emotion90.append(len(idata9.loc[idata9['label3']==0,:])/ilen9)
        topic_emotion91.append(len(idata9.loc[idata9['label3']==1,:])/ilen9)
        topic_emotion92.append(len(idata9.loc[idata9['label3']==2,:])/ilen9)
        
x=pd.date_range('2/1/2020','4/29/2020')
'''
message_dic = {"date":x,"消极占比" : timeseries_emotion0,"中立占比" : timeseries_emotion1,"积极占比" : timeseries_emotion2,
               "r_topic0" : timeseries_topic0,"r_topic1" : timeseries_topic1,"r_topic2" : timeseries_topic2,
               "r_topic3" : timeseries_topic3,"r_topic4" : timeseries_topic4,"r_topic5" : timeseries_topic5,
               "r_topic6" : timeseries_topic6,"r_topic7" : timeseries_topic7,"r_topic8" : timeseries_topic8,
               "r_topic9" : timeseries_topic9}
df = pd.DataFrame.from_dict(message_dic)
df.to_csv("E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//00 推特结果分析//twitter_resultAll.csv",encoding='utf-8',index=False)
'''
tpe0=[topic_emotion02[i]+topic_emotion00[i]*(-1) for i in range(len(topic_emotion00))]
tpe1=[topic_emotion12[i]+topic_emotion10[i]*(-1) for i in range(len(topic_emotion10))]
tpe2=[topic_emotion22[i]+topic_emotion20[i]*(-1) for i in range(len(topic_emotion20))]
tpe3=[topic_emotion32[i]+topic_emotion30[i]*(-1) for i in range(len(topic_emotion30))]
tpe4=[topic_emotion42[i]+topic_emotion40[i]*(-1) for i in range(len(topic_emotion40))]
tpe5=[topic_emotion52[i]+topic_emotion50[i]*(-1) for i in range(len(topic_emotion50))]
tpe6=[topic_emotion62[i]+topic_emotion60[i]*(-1) for i in range(len(topic_emotion60))]
tpe7=[topic_emotion72[i]+topic_emotion70[i]*(-1) for i in range(len(topic_emotion70))]
tpe8=[topic_emotion82[i]+topic_emotion80[i]*(-1) for i in range(len(topic_emotion80))]
tpe9=[topic_emotion92[i]+topic_emotion90[i]*(-1) for i in range(len(topic_emotion90))]
'''
message_dic2 = {"date":x,"s_topic0" : tpe0,"s_topic1" : tpe1,"s_topic2" : tpe2,"s_topic3" : tpe3,"s_topic4" : tpe4,"s_topic5" : tpe5,
               "s_topic6" : tpe6,"s_topic7" : tpe7,"s_topic8" : tpe8,"s_topic9" : tpe9}
df2 = pd.DataFrame.from_dict(message_dic2)
df2.to_csv("E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//00 推特结果分析//twitter_tpe.csv",encoding='utf-8',index=False)
'''
# 3. 画图
import matplotlib
matplotlib.rc("font",family='YouYuan')
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams.update({'font.size':10})
'''
plt.plot(x,tpe0,label=u'话题0 学习出行',color='blueviolet')
plt.plot(x,tpe1,label=u'话题1 居家隔离',color='orange')
plt.plot(x,tpe2,label=u'话题2 其他',color='rosybrown')
plt.plot(x,tpe3,label=u'话题3 共同战疫',color='thistle')
plt.plot(x,tpe4,label=u'话题4 封锁下的市场',color='darkseagreen')
plt.plot(x,tpe5,label=u'话题5 防护治疗',color='mediumaquamarine')
plt.plot(x,tpe6,label=u'话题6 对领导政府的评价',color='darkturquoise')
plt.plot(x,tpe7,label=u'话题7 国际疫情状况',color='salmon')
plt.plot(x,tpe8,label=u'话题8 美国检测确诊情况',color='wheat')
plt.plot(x,tpe9,label=u'话题9 情绪表达',color='powderblue')
plt.rcParams.update({'font.size':7})
plt.legend(loc='lower right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('话题情绪得分变化图', fontproperties=font_set, fontsize=20)
plt.savefig('./话题情绪得分变化图')
plt.show()
'''

plt.plot(x,counts_twitter,c='mediumpurple')
plt.grid()
plt.title('twitter数目变化图', fontproperties=font_set, fontsize=20)
plt.savefig('./twitter数目变化图')
plt.show()
'''
plt.plot(x,timeseries_emotion0,c='turquoise',label=u'消极')
plt.plot(x,timeseries_emotion1,c='bisque',label=u'中立')
plt.plot(x,timeseries_emotion2,c='darkseagreen',label=u'积极')'''

plt.plot(x,timeseries_emotion0,label=u'消极',color='mediumpurple')
plt.plot(x,timeseries_emotion1,label=u'中立',color='burlywood')
plt.plot(x,timeseries_emotion2,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('推特宏观情感占比变化图', fontproperties=font_set, fontsize=20)
plt.savefig('./宏观情感占比变化图')
plt.show()
'''
labels=[u'话题0 学习出行',u'话题1 居家隔离',u'话题2 其他',u'话题3 共同战疫',u'话题4 封锁下的市场',u'话题5 防护治疗',
        u'话题6 对领导政府的评价',u'话题7 国际疫情状况',u'话题8 美国检测确诊情况',u'话题9 情绪表达']
colors = ['tan','navajowhite','lightsalmon','coral','tomato','lightcoral','powderblue','mediumturquoise','lightseagreen','cadetblue']
plt.pie(ctp,labels=labels,autopct='%1.2f%%',colors=colors)
plt.title('各话题总体占比饼图', fontproperties=font_set, fontsize=20)
plt.savefig('./宏各话题总体占比饼图')
plt.show()

plt.plot(x,timeseries_topic0,label=u'话题0 学习出行',color='blueviolet')
plt.plot(x,timeseries_topic1,label=u'话题1 居家隔离',color='orange')
plt.plot(x,timeseries_topic2,label=u'话题2 其他',color='gainsboro')
plt.plot(x,timeseries_topic3,label=u'话题3 共同战疫',color='thistle')
plt.plot(x,timeseries_topic4,label=u'话题4 封锁下的市场',color='darkseagreen')
plt.plot(x,timeseries_topic5,label=u'话题5 防护治疗',color='beige')
plt.plot(x,timeseries_topic6,label=u'话题6 对领导政府的评价',color='darkturquoise')
plt.plot(x,timeseries_topic7,label=u'话题7 国际疫情状况',color='salmon')
plt.plot(x,timeseries_topic8,label=u'话题8 美国检测确诊情况',color='wheat')
plt.plot(x,timeseries_topic9,label=u'话题9 情绪表达',color='powderblue')
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('宏观话题占比变化图', fontproperties=font_set, fontsize=20)
plt.savefig('./宏观话题占比变化图')
plt.show()
'''
plt.plot(x,topic_emotion00,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion01,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion02,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()#显示网格线
plt.title('话题0"学习出行(tweet)"情绪变化图', fontproperties=font_set, fontsize=20)
plt.savefig('./话题0情绪变化图')
plt.show()

plt.plot(x,topic_emotion10,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion11,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion12,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题1"居家隔离"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题1情绪变化图')
plt.show()

plt.plot(x,topic_emotion20,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion21,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion22,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题2"其他"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题2情绪变化图')
plt.show()

plt.plot(x,topic_emotion30,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion31,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion32,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题3"共同战疫"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题3情绪变化图')
plt.show()

plt.plot(x,topic_emotion40,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion41,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion42,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题4"封锁下的市场"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题4情绪变化图')
plt.show()

plt.plot(x,topic_emotion50,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion51,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion52,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题5"防护治疗"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题5情绪变化图')
plt.show()

plt.plot(x,topic_emotion60,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion61,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion62,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题6"对领导政府的评价"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题6情绪变化图')
plt.show()

plt.plot(x,topic_emotion70,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion71,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion72,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题7"国际疫情状况"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题7情绪变化图')
plt.show()

plt.plot(x,topic_emotion80,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion81,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion82,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')
plt.grid()
plt.title('话题8"美国检测确诊情况"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题8情绪变化图')
plt.show()
            
plt.plot(x,topic_emotion90,label=u'消极',color='mediumpurple')
plt.plot(x,topic_emotion91,label=u'中立',color='burlywood')
plt.plot(x,topic_emotion92,label=u'积极',color='mediumturquoise')
plt.legend(loc='upper right')#标签显示位置upper right
plt.grid()
plt.title('话题9"情绪表达"情绪变化图(tweet)', fontproperties=font_set, fontsize=20)
plt.savefig('./话题9情绪变化图')
plt.show()

