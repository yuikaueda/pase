import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize

#plt.savefig("ini.png")

df_ini = pd.read_table('ini_c.dat')
df_ini_pivot = pd.pivot_table(data=df_ini, values='c', columns='x', index='y', aggfunc=np.mean)

#print(df)

fig1, ax1 = plt.subplots(1, 1)
mappable1 = ax1.pcolor(df_ini_pivot.columns, df_ini_pivot.index, df_ini_pivot.T)
fig1.colorbar(mappable1, ax = ax1)
fig1.tight_layout()
#plt.show()
fig1.savefig("2ini.png")  

df = pd.read_table('result2400.dat')
df_pivot = pd.pivot_table(data=df, values='c', columns='x', index='y', aggfunc=np.mean)

#print(df)

fig2, ax2 = plt.subplots(1, 1)
mappable2 = ax2.pcolor(df_pivot.columns, df_pivot.index, df_pivot.T)
fig2.colorbar(mappable2, ax = ax2)
fig2.tight_layout()
#plt.show()
fig2.savefig("2step2400.png")  
