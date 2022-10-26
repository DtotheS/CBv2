import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import csv

df = pd.read_csv("/Users/agathos/DtotheS/CBv2/data/result100n.csv")
df2 = pd.read_csv("/Users/agathos/DtotheS/CBv2/data/result100t.csv")
df = df[2:].reset_index(drop=True)
df2 = df2[2:].reset_index(drop=True)

rdf = df[df['condition']=='1'] # 1 real
fdf = df[df['condition']=='2'] # 2 fake

rdft = df2[df2['condition']=='1'] # text answers
fdft = df2[df2['condition']=='2']

col = df.columns.to_list()
# col.index('Q3')
cdemo = col[17:24]
creal = col[24:174:5] # len = 30
cfake = col[174:324:5] # len = 30
code = 'mTurkcode'

# column selection by topic, veracity, authors' news leaning
a_r_l = col[24:49:5]
a_r_r = col[49:74:5]
g_r_l = col[74:99:5]
g_r_r = col[99:124:5]
i_r_l = col[124:149:5]
i_r_r = col[149:174:5]
a_f_l = col[174:199:5]
a_f_r = col[199:224:5]
g_f_l = col[224:249:5]
g_f_r = col[249:274:5]
i_f_l = col[274:299:5]
i_f_r = col[299:324:5]

# 2 News Selection based on the average partisanship score (gap from 3)
arl = [a_r_l[0],a_r_l[1]]
arr = [a_r_r[0],a_r_r[4]]
grl = [g_r_l[3],g_r_l[4]]
grr = [g_r_r[0],g_r_r[1]]
irl = [i_r_l[2],i_r_l[3]]
irr = [i_r_r[0],i_r_r[1]]
afl = [a_f_l[2],a_f_l[3]]
afr = [a_f_r[1],a_f_r[2]]
gfl = [g_f_l[0],g_f_l[1]]
gfr = [g_f_r[2],g_f_r[3]]
ifl = [i_f_l[2],i_f_l[3]]
ifr = [i_f_r[1],i_f_r[2]]
# foo = 24
# for i in range(12):
#     foo += 25
#     print(foo)

result = pd.DataFrame()
## Average partisanship score
# (rdf['mTurkcode'] == rdft['mTurkcode']).sum()
# (fdf['mTurkcode'] == fdft['mTurkcode']).sum()
result['mturkcode'] = rdf[code].astype(int).append(fdf[code].astype(int))
result[['gender','age','edu','ethnicity','pinterest','pcynical','pstance']] = rdft[cdemo].append(fdft[cdemo])
result['veracity(1real)'] = rdf['condition'].astype(int).append(fdf['condition'].astype(int))

# rdf[arl].astype(int).mean(axis=1).mean()
# fdf[afl].astype(int).mean(axis=1).mean()
al_mean = rdf[arl].astype(int).mean(axis=1).append(fdf[afl].astype(int).mean(axis=1))
ar_mean = rdf[arr].astype(int).mean(axis=1).append(fdf[afr].astype(int).mean(axis=1))
gl_mean = rdf[grl].astype(int).mean(axis=1).append(fdf[gfl].astype(int).mean(axis=1))
gr_mean = rdf[grr].astype(int).mean(axis=1).append(fdf[gfr].astype(int).mean(axis=1))
il_mean = rdf[irl].astype(int).mean(axis=1).append(fdf[ifl].astype(int).mean(axis=1))
ir_mean = rdf[irr].astype(int).mean(axis=1).append(fdf[ifr].astype(int).mean(axis=1))

li1 = ['a-lib-mean','a-rep-mean','g-lib-mean','g-rep-mean','i-lib-mean','i-rep-mean']
li2 = [al_mean,ar_mean,gl_mean,gr_mean,il_mean,ir_mean]
for i in range(len(li1)):
    result[li1[i]] = li2[i]

# partisanship: gap from 3 : liberal = 3 - mean score // conservative = mean score - 3. positive means correctly labeled by participants.
li3 = ['a-lib-dist','a-rep-dist','g-lib-dist','g-rep-dist','i-lib-dist','i-rep-dist']
for i in range(len(li1)):
    if i%2 == 0:
        result[li3[i]] = 3 - result[li1[i]]
    else:
        result[li3[i]] = result[li1[i]] - 3

result[li3].mean()

# result.to_csv("/Users/agathos/DtotheS/CBv2/data/result_clean_wide.csv",index=False)

## Repeated measure ANOVA
!pip install statsmodels
from statsmodels.stats.anova import AnovaRM
print(AnovaRM(data=result, depvar='response', subject='patient', within=['drug']).fit())