import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.interpolate import interp1d
import pandas as pd
#plt.style.use('seaborn-dark-palette')

def callmain(file,ylab):
    plt.figure()
    ax = plt.axes()

    #file="4"
    with open(file+".csv", 'rt')as f:   #  4----5----6
        content = csv.reader(f)
        # Store data in array
        data = []
        for row in content:
            tem = []
            for j in row:
                tem.append((float(j)))  # attribute r from each row
            data.append(tem)
        #data = np.transpose(data)



    # to plot graph
    def plot_graph(result_1, result_2, result_3, result_4):
        loc, result = [], []
        result.append(result_1)  # appending the result
        result.append(result_2)
        result.append(result_3)
        result.append(result_4)


        result = np.transpose(result)

        # labels for bars
        labels=['CybS-CC-SACGAN-COA','RF-RBFNN','FEFS-DLM','IMFL-IDSCS','HSBOA_QRNN','Proposed TaWSA_WRN']

        tick_labels = [5,10,15,20]  # metrics with iteration = 20


        bar_width, s = 0.10, 0.0  # bar width, space between bars
        for i in range(len(result)):  # allocating location for bars
            if i is 0:  # initial location - 1st result
                tem = []
                for j in range(len(tick_labels)):
                    tem.append(j + 1)
                loc.append(tem)
            else:  # location from 2nd result
                tem = []
                for j in range(len(loc[i - 1])):
                    tem.append(loc[i - 1][j] + s + bar_width)
                loc.append(tem)
        color1=['#C0C0C0','#DDA0DD','#73C2FB', '#FDBCB4','#0095B6','#E3F988']

        #color1=['#9F8170','#E9D66B','#F08080', '#FDBCB4','#ACE1AF','#B284BE','#0095B6']

        # plotting a bar chart
        for i in range(len(result)):
            plt.bar(loc[i], result[i],label=labels[i], tick_label=tick_labels, edgecolor='black',color=color1[i],width=bar_width)       #,,label=labels[i],

        plt.legend()  # show a legend on the plot
        font = font_manager.FontProperties(family='Calibri',  # 'Times new roman',
                                           weight='bold',
                                           style='normal', size=11)
        plt.xlabel("Feature dimension", fontweight='bold', size=12) #  Training data(%)

        plt.ylabel(ylab, fontweight='bold', size=12) ####  Precision(%)  ,  Recall(%)   , F Measure(%)
        plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.16),
                   fancybox=True, ncol=3, fontsize=10,prop=font)

        plt.savefig(file+'.svg',dpi=900)
        #MMRE,MdMRE,Speed(s),RMSE,Pred,Model size,MAE,MSE,RMSE,Prediction error
    result_1, result_2, result_3, result_4= data[0], data[1], data[2], data[3]
    plot_graph(result_1, result_2, result_3, result_4)
    plt.show()

callmain("graph/1","Accuracy(%)")
callmain("graph/2","TPR(%)")
callmain("graph/3","TNR(%)")

def callmain(file,ylab):
    #file='7'   #  1---2----3
    data = pd.read_csv(file+'.csv',header=None)
    data=np.array(data)

    # Define x, y, and xnew to resample at.

    x = [10,20,30,40]

    linestyles = ['--', '-.', '-', 'dotted', (5, (10, 3)), (0, (5, 10)), (0, (5, 1))]
    markers_list = ['o', '^', 's', '*', 'p']

    colors1 = ['cyan', 'purple', 'green','orange','red']
    colors2 = ['yellow', 'red', 'orange','green','blue']

    colors1=['#73C2FB','#E0B0FF','#9AB973','#FDBCB4','#FF7F7F']
    colors2=['#FBEC5D','#E5AA70','#CC8899','#40E0D0','#1E90FF']
    #colors2.reverse()
    new_labels=['CSO+WRN','GSO+WRN','TS+WRN','WSA+WRN','Proposed TaWSA+WRN']
    # plot
    for i in range(len(data[0])):
        y = data[:,i]
        xnew = np.linspace(x[0], x[-1], num=200, endpoint=True)
        #xnew = np.linspace(20, 80, num=200, endpoint=True)
        f_cubic = interp1d(x, y, kind='cubic')
        plt.plot(xnew, f_cubic(xnew), linestyle='-',color=colors1[i])
        plt.plot(x, y, 'o', marker=markers_list[i], markersize=12, markerfacecolor=colors2[i], markeredgecolor="black",
                 alpha=0.9, label=new_labels[i])
    font = font_manager.FontProperties(family='Calibri',  # 'Times new roman',
                                           weight='bold',
                                           style='normal', size=11)
    plt.xticks(x)
    plt.grid(color='0.9')
    plt.xlabel('Swarm size',fontweight='bold', size=12)  #######  Training data(%)
    plt.ylabel(ylab,fontweight='bold', size=12)   #### Precision(%)  ,  Recall(%)  ,  F1 Score(%)
    plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.16),
                   fancybox=True, ncol=3, fontsize=8,prop=font)
    plt.savefig(file+'.svg',dpi=900)
    plt.show()

callmain("graph/4","Accuracy(%)")
callmain("graph/5","TPR(%)")
callmain("graph/6","TNR(%)")


def callmain(file,ylab):
    plt.figure()
    ax = plt.axes()

    #file="4"
    with open(file+".csv", 'rt')as f:   #  4----5----6
        content = csv.reader(f)
        # Store data in array
        data = []
        for row in content:
            tem = []
            for j in row:
                tem.append((float(j)))  # attribute r from each row
            data.append(tem)
        #data = np.transpose(data)



    # to plot graph
    def plot_graph(result_1, result_2, result_3, result_4):
        loc, result = [], []
        result.append(result_1)  # appending the result
        result.append(result_2)
        result.append(result_3)
        result.append(result_4)


        result = np.transpose(result)

        # labels for bars
        labels=['CybS-CC-SACGAN-COA','RF-RBFNN','FEFS-DLM','IMFL-IDSCS','HSBOA_QRNN','Proposed TaWSA_WRN']

        tick_labels = [5,6,7,8]  # metrics with iteration = 20


        bar_width, s = 0.10, 0.0  # bar width, space between bars
        for i in range(len(result)):  # allocating location for bars
            if i is 0:  # initial location - 1st result
                tem = []
                for j in range(len(tick_labels)):
                    tem.append(j + 1)
                loc.append(tem)
            else:  # location from 2nd result
                tem = []
                for j in range(len(loc[i - 1])):
                    tem.append(loc[i - 1][j] + s + bar_width)
                loc.append(tem)
        #color1=['#C0C0C0','#DDA0DD','#73C2FB', '#FDBCB4','#0095B6','#E3F988']

        color1=['#9F8170','#E9D66B','#F08080', '#FDBCB4','#ACE1AF','#B284BE','#0095B6']

        # plotting a bar chart
        hatch1=['//','//','//','//','//','o']
        for i in range(len(result)):
            plt.bar(loc[i], result[i],label=labels[i], tick_label=tick_labels, edgecolor='black',color=color1[i],width=bar_width,hatch=hatch1[i])       #,,label=labels[i],

        plt.legend()  # show a legend on the plot
        font = font_manager.FontProperties(family='Calibri',  # 'Times new roman',
                                           weight='bold',
                                           style='normal', size=11)
        plt.xlabel("K Value", fontweight='bold', size=12) #  Training data(%)

        plt.ylabel(ylab, fontweight='bold', size=12) ####  Precision(%)  ,  Recall(%)   , F Measure(%)
        plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.16),
                   fancybox=True, ncol=3, fontsize=10,prop=font)

        plt.savefig(file+'.svg',dpi=900)
        #MMRE,MdMRE,Speed(s),RMSE,Pred,Model size,MAE,MSE,RMSE,Prediction error
    result_1, result_2, result_3, result_4= data[0], data[1], data[2], data[3]
    plot_graph(result_1, result_2, result_3, result_4)
    plt.show()

callmain("graph/7","Accuracy(%)")
callmain("graph/8","TPR(%)")
callmain("graph/9","TNR(%)")


def callmain(file,ylab):
    #file='13'
    data= pd.read_csv(file+".csv",header=None)  # 11---12----13
    data=np.array(data)
    fig = plt.figure()
    plt.subplot(111, polar=True)
    legend_str=['Proposed TaWSA_WRN With Epoch=10','Proposed TaWSA_WRN With Epoch=20','Proposed TaWSA_WRN With Epoch=30','Proposed TaWSA_WRN With Epoch=40']
    index = np.arange(1, 5, 1)*(1.9*np.pi/4)
    bar_width = 0.15
    opacity = 1
    xpos = np.array([1, 2, 3,4])*(2*np.pi/4)
    xlabels=['60','70','80','90']
    #color=['cyan','pink','lightgreen','red']
    color=['#73C2FB', '#FDBCB4','#0095B6','#E3F988']
    plt.bar(index,                  height=data[:, 0], width=(2*np.pi/(6*10)), bottom=0,color=color[0])
    plt.bar(index+(2*np.pi/(6*10)), height=data[:, 1], width=(2*np.pi/(6*10)), bottom=0,color=color[1])
    plt.bar(index+2*(2*np.pi/(6*10)), height=data[:, 2], width=(2*np.pi/(6*10)), bottom=0,color=color[2])
    plt.bar(index+3*(2*np.pi/(6*10)), height=data[:, 3], width=(2*np.pi/(6*10)), bottom=0,color=color[3])
    #plt.bar(index+4*(2*np.pi/(6*10)), height=data[:, 4], width=(2*np.pi/(6*10)), bottom=10,color='purple')
    font = font_manager.FontProperties(family='Calibri',  # 'Times new roman',
                                           weight='bold',
                                           style='normal', size=10)
    ax = plt.gca()
    leg=ax.legend(legend_str, loc='right',bbox_to_anchor=(1.34, 0.9),fontsize=7,prop=font)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.set_alpha(1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    plt.xlabel('Learning data(%)',fontweight='bold', size=12)  # Training data(%)

    plt.title(ylab,fontweight='bold', size=12)  # Precision(%)  ,  Recall(%)   , F Measure(%)
    plt.savefig(file+'.svg',dpi=900)
    plt.show()

callmain("graph/10","Accuracy(%)")
callmain("graph/11","TPR(%)")
callmain("graph/12","TNR(%)")