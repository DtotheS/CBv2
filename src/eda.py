import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns

plt.style.use('seaborn-bright')

df = pd.read_csv("/Users/agathos/DtotheS/CBv2/data/result100n.csv")
df2 = pd.read_csv("/Users/agathos/DtotheS/CBv2/data/result100t.csv")
df = df[2:].reset_index(drop=True)
df2 = df2[2:].reset_index(drop=True)

rdf = df[df['condition']=='1']
fdf = df[df['condition']=='2']

rdft = df2[df2['condition']=='1']
fdft = df2[df2['condition']=='2']

col = df.columns.to_list()
# col.index('Q3')
cdemo = col[17:24]
creal = col[24:174:5] # len = 30
cfake = col[174:324:5] # len = 30


result = pd.DataFrame()
## Average partisanship score
result['score_mean'] = rdf[creal].astype(int).mean().append(fdf[cfake].astype(int).mean())

## 1,2, counts
lib_count = []
mod_count = []
con_count = []
for news in creal:
    c = Counter(rdf[news].astype(int).to_list())
    lib_count.append(c[1] + c[2])
    mod_count.append(c[3])
    con_count.append(c[4] + c[5])

for news in cfake:
    c = Counter(fdf[news].astype(int).to_list())
    lib_count.append(c[1] + c[2])
    mod_count.append(c[3])
    con_count.append(c[4] + c[5])

result['lib_count'] = lib_count
result['con_count'] = con_count
result['mod_count'] = mod_count

for i in range(30):
    print(result['lib_count'][i] + result['con_count'][i] + result['mod_count'][i] == 49)
for i in range(30,len(result)):
    print(result['lib_count'][i] + result['con_count'][i] + result['mod_count'][i] == 51)

# Check majority label of participants
part_label = []
for i in range(len(result)):
    if i < 30:
        cri = 24
    else:
        cri = 25

    if result['lib_count'][i] > cri:
        part_label.append("liberal")
    elif result['con_count'][i] > cri:
        part_label.append("conservative")
    else:
        part_label.append("fail to achieve half agreement")

# Veracity
ver = []
for i in range(30):
    ver.append("real")
for i in range(30):
    ver.append("fake")
result['veracity'] = ver

# Topic
topic = []
for i in range(len(result)):
    if i<10:
        topic.append("abortion")
    elif i<20:
        topic.append("gun")
    elif i <30:
        topic.append("inflation")
    elif i < 40:
        topic.append("abortion")
    elif i < 50:
        topic.append("gun")
    elif i < 60:
        topic.append("inflation")
result['topic'] = topic

# Authors' initial label
our_label = []
for i in range(len(result)):
    if (i // 5)%2 == 0:
        our_label.append("liberal")
    else:
        our_label.append("conservative")

result['authors_label'] = our_label

# News id
news_num = []
news_num = [x+1 for x in range(10)]
news_num = news_num * 6
result['news_number'] = news_num

# part vs. author label match
result['label_match'] = False
for i in range(len(result)):
    if result['part_majority'][i] == result['authors_label'][i]:
        result['label_match'][i] = True

# gap from 3
delta = []
for i in range(len(result)):
    if result['authors_label'][i] == 'liberal':
        gap = 3 - result['score_mean'][i]
    else:
        gap = result['score_mean'][i] - 3
    delta.append(gap)

result['distance_from_3'] = delta

## Distance graph
# Real - Abortion
x = []
for i in range(10):
    x.append(result['authors_label'][i]+str(result['news_number'][i]))
y = result['distance_from_3'][:10].to_list()
x
plt.bar(x,y, label="Distance from 3")
for i in range(len(x)):
    plt.text(x[i], y[i], round(y[i],1), ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Distance from neutral (3): Real-Abortion")
plt.ylabel("Lib: 3-mean_score / Con: mean_score-3")
plt.xlabel("News")
plt.ylim([-1,2])
plt.savefig("/Users/agathos/DtotheS/CBv2/img/real-abortion.png",dpi=600)
plt.show()
plt.close()

# Real - gun
rs = 10
re = 20
x = []
for i in range(rs,re):
    x.append(result['authors_label'][i]+str(result['news_number'][i]))
y = result['distance_from_3'][rs:re].to_list()
x
plt.bar(x,y, label="Distance from 3")
for i in range(len(x)):
    plt.text(x[i], y[i], round(y[i],1), ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Distance from neutral (3): Real-Gun")
plt.ylabel("Lib: 3-mean_score / Con: mean_score-3")
plt.xlabel("News")
plt.ylim([-1,2])
plt.savefig("/Users/agathos/DtotheS/CBv2/img/real-gun.png",dpi=600)
plt.show()
plt.close()

# Real - infla
rs = 20
re = 30
x = []
for i in range(rs,re):
    x.append(result['authors_label'][i]+str(result['news_number'][i]))
y = result['distance_from_3'][rs:re].to_list()
x
plt.bar(x,y, label="Distance from 3")
for i in range(len(x)):
    plt.text(x[i], y[i], round(y[i],1), ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Distance from neutral (3): Real-Inflation")
plt.ylabel("Lib: 3-mean_score / Con: mean_score-3")
plt.xlabel("News")
plt.ylim([-1,2])
plt.savefig("/Users/agathos/DtotheS/CBv2/img/real-inflation.png",dpi=600)
plt.show()
plt.close()

# Fake - abortion
rs = 30
re = 40
x = []
for i in range(rs,re):
    x.append(result['authors_label'][i]+str(result['news_number'][i]))
y = result['distance_from_3'][rs:re].to_list()
x
plt.bar(x,y, label="Distance from 3")
for i in range(len(x)):
    plt.text(x[i], y[i], round(y[i],1), ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Distance from neutral (3): Fake-Abortion")
plt.ylabel("Lib: 3-mean_score / Con: mean_score-3")
plt.xlabel("News")
plt.ylim([-1,2])
plt.savefig("/Users/agathos/DtotheS/CBv2/img/fake-abortion.png",dpi=600)
plt.show()
plt.close()

# Fake - gun
rs = 40
re = 50
x = []
for i in range(rs,re):
    x.append(result['authors_label'][i]+str(result['news_number'][i]))
y = result['distance_from_3'][rs:re].to_list()
x
plt.bar(x,y, label="Distance from 3")
for i in range(len(x)):
    plt.text(x[i], y[i], round(y[i],1), ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Distance from neutral (3): Fake-Gun")
plt.ylabel("Lib: 3-mean_score / Con: mean_score-3")
plt.xlabel("News")
plt.ylim([-1,2])
plt.savefig("/Users/agathos/DtotheS/CBv2/img/fake-gun.png",dpi=600)
plt.show()
plt.close()

# Fake - inflation
rs = 50
re = 60
x = []
for i in range(rs,re):
    x.append(result['authors_label'][i]+str(result['news_number'][i]))
y = result['distance_from_3'][rs:re].to_list()

plt.bar(x,y, label="Distance from 3")
for i in range(len(x)):
    plt.text(x[i], y[i], round(y[i],1), ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.title("Distance from neutral (3): Fake-Inflation")
plt.ylabel("Lib: 3-mean_score / Con: mean_score-3")
plt.xlabel("News")
plt.ylim([-1,2])
plt.savefig("/Users/agathos/DtotheS/CBv2/img/fake-inflation.png",dpi=600)
plt.show()
plt.close()

result.to_csv("/Users/agathos/DtotheS/CBv2/data/result_clean.csv")


''' Demo Graph
# df2.Q3.value_counts().plot(kind='bar')
def my_fmt(x):
    # print(x)
    return '{:.1f}%\n({:.0f})'.format(x, len(df)*x/100)

cdemo

## Condition
x = df.groupby('condition').count().index.to_list()
y = df.groupby('condition').count()['StartDate'].to_list()
x = ['Real News','Fake News']
plt.pie(y,labels = x, autopct='%.0f')
# for i in range(len(x)):
#     plt.text(x[i], y[i], yp[i], ha='center')
# plt.xticks(np.arange(len(x)), x, rotation=90)
# plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Participants Assignment (Total: 100)")
# plt.ylabel("Number of participants (total: %s)" % len(df))
# plt.xlabel("Gender")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/condition.png",dpi=600)
plt.show()
plt.close()

## Gender
x = df2.groupby([cdemo[0]])['condition'].count().index.to_list()
y = df2.groupby([cdemo[0]])['condition'].count().to_list()
plt.pie(y,labels = x, autopct='%.0f')
# for i in range(len(x)):
#     plt.text(x[i], y[i], yp[i], ha='center')
# plt.xticks(np.arange(len(x)), x, rotation=90)
# plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Gender")
# plt.ylabel("Number of participants (total: %s)" % len(df))
# plt.xlabel("Gender")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/gender.png",dpi=600)
plt.show()
plt.close()

cdemo[1]
## Age
x = df2.groupby([cdemo[1]])['condition'].count().index.to_list()
y = df2.groupby([cdemo[1]])['condition'].count().to_list()
plt.pie(y,labels = x, autopct='%.0f')
# for i in range(len(x)):
#     plt.text(x[i], y[i], yp[i], ha='center')
# plt.xticks(np.arange(len(x)), x, rotation=90)
# plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Age")
# plt.ylabel("Number of participants (total: %s)" % len(df))
# plt.xlabel("Gender")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/age.png",dpi=600)
plt.show()
plt.close()

cdemo[2]
## Edu
x = df2.groupby([cdemo[2]])['condition'].count().index.to_list()
y = df2.groupby([cdemo[2]])['condition'].count().to_list()
plt.pie(y,labels = x, autopct='%.0f')
# for i in range(len(x)):
#     plt.text(x[i], y[i], yp[i], ha='center')
# plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(left=0.4, right=0.8)
plt.title("Education")
# plt.ylabel("Number of participants (total: %s)" % len(df))
# plt.xlabel("Gender")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/edu.png",dpi=600)
plt.show()
plt.close()

cdemo[3]
## Ethnicity
x = df2.groupby([cdemo[3]])['condition'].count().index.to_list()
y = df2.groupby([cdemo[3]])['condition'].count().to_list()
# yp = []
# for i in y:
#     yp.append(str(int(round(i/len(df) * 100,1)))+"%" + "("+str(i)+")")
x[4] = 'Native American ...'
# plt.pie(y,labels = x, autopct=my_fmt)
plt.bar(x,y, label="Ethnicity")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(top=0.9,bottom=0.4)
plt.title("Ethnicity")
plt.ylabel("Number of participants (total: %s)" % len(df))
plt.xlabel("Ethnicity")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/ethnicity.png",dpi=600)
plt.show()
plt.close()

## I am intersted in politics
x = df2.groupby([cdemo[4]])['condition'].count().index.to_list()
y = df2.groupby([cdemo[4]])['condition'].count().to_list()
x = [x[3],x[0],x[1],x[4],x[2]]
y = [y[3],y[0],y[1],y[4],y[2]]
# yp = []
# for i in y:
#     yp.append(str(int(round(i/len(df) * 100,1)))+"%" + "("+str(i)+")")

# plt.pie(y,labels = x, autopct=my_fmt)
plt.bar(x,y, label="interested in politics")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(top=0.9,bottom=0.4)
plt.title("I am interested in politics")
plt.ylabel("Number of participants (total: %s)" % len(df))
# plt.xlabel("Ethnicity")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/interest.png",dpi=600)
plt.show()
plt.close()

## I am politically cynical
x = df2.groupby([cdemo[5]])['condition'].count().index.to_list()
y = df2.groupby([cdemo[5]])['condition'].count().to_list()

x = [x[3],x[0],x[1],x[4],x[2]]
y = [y[3],y[0],y[1],y[4],y[2]]
# yp = []
# for i in y:
#     yp.append(str(int(round(i/len(df) * 100,1)))+"%" + "("+str(i)+")")

# plt.pie(y,labels = x, autopct=my_fmt)
plt.bar(x,y, label="politically cynical")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(top=0.9,bottom=0.4)
plt.title("I am politically cynical")
plt.ylabel("Number of participants (total: %s)" % len(df))
# plt.xlabel("Ethnicity")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/cynical.png",dpi=600)
plt.show()
plt.close()

## Political Stance
x = df2.groupby([cdemo[6]])['condition'].count().index.to_list()
y = df2.groupby([cdemo[6]])['condition'].count().to_list()

x = [x[4],x[1],x[2],x[0],x[3]]
y = [y[4],y[1],y[2],y[0],y[3]]
# yp = []
# for i in y:
#     yp.append(str(int(round(i/len(df) * 100,1)))+"%" + "("+str(i)+")")

# plt.pie(y,labels = x, autopct=my_fmt)
plt.bar(x,y, label="Political Stance")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha='center')
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(top=0.9,bottom=0.4)
plt.title("Self-identified political stance")
plt.ylabel("Number of participants (total: %s)" % len(df))
# plt.xlabel("Ethnicity")
plt.savefig("/Users/agathos/DtotheS/CBv2/img/political_stance.png",dpi=600)
plt.show()
plt.close()
'''
