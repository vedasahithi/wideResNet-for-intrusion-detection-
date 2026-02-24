import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
plt.figure()
style.use('bmh') # seaborn , ggplot , bmh , fivethirtyeight

#ax.set_facecolor('#ffd0f4')
file="sel-fea"
data=pd.read_csv(file+".csv")
data=np.array(data)

legends1 = ["serror_rate","rerror_rate","same_srv_rate","diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_srv_serror_rate","dst_host_srv_rerror_rate"]  # x-axis labels

marker = ['x', '*',".",">","o"]
#labels =  [4,1,0,2,3] # metrics
labels2 =  [7,6,5,4,3,2,1,0] # metrics


color_code = ['#9F8170','#E9D66B','#F08080', '#FDBCB4','#ACE1AF','#B284BE','#0095B6','#E3F988']

l=[i for i in range(len(data))]
plt.subplots_adjust(top=0.81)
# Plotting the result
for i in range(len(data[0])):
    print('i :',i)
    plt.scatter(l,data[:,labels2[i]],  label=legends1[labels2[i]], color=color_code[i],edgecolors='black',linewidth=0.01)

plt.xlabel('Number of samples', fontweight='bold', size=10)

plt.ylabel('Feature values', fontweight='bold', size=10)
plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.26), fancybox=True, ncol=3, fontsize=9.3)    # show a legend on the plot -- here legends are metrics
plt.savefig(file + '.jpg',dpi=900)  # to show the plot
plt.show()
