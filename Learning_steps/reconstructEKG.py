# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:36:35 2016

python code to interact with EKG data from PDF scans or hold

@author: narinder
"""

TESTFILE = "output/Brigham_Sample_EKG_full.csv"
LEADS = ['Lead I', 'Lead aVR', 'Lead V1', 'Lead V4', 'Lead II', 'Lead aVL','Lead V2', 'Lead V5', 'Lead III', 'Lead aVF', 'Lead V3', 'Lead V6']
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

'''
    This is loading the csv file outputted by the EKG Tool.  It expects
    a column called "Lead" with the potential values shown in LEADS and a column called
    actual_Y representing the y position of the line relative to the starting axis.
'''
def loadScanCSV(filename=TESTFILE):
    df = pd.read_csv(filename,header=(0),skipinitialspace=True)
    df.dropna()
    return df

'''
    plots EKG leads on a grap w/number of columns specificied for
    the leads passed in with a record (time) limit.  In this version
    the times for each lead are not necessarily the same (ie start / stop at diff
    points)
    if graphics option is set to Automatic, it will pop out the graph
    #goal is to basically reproduce what an EKG looks like
'''
def plotRecordTimeDelta(df, leads=LEADS, columns=4,lengthLimit=None, patient=None):

    fig, axes = plt.subplots(nrows=3, ncols=4, sharey='row', sharex='col')
    plt.subplots_adjust(wspace=0, hspace=0)

    '''to make graphing simple, the below is not using any value for the X axis
        its just assuming they occur every 1/300th of a second. Then the grid lines
        are set at 1/5 of a second for major intervals and 1/25s for minor.
        The only misleading part of this is that the actual sub graphs are slightly time shifted.
        I may change this later, but the visual output would look the same.
    '''
    col=0
    row=0
    for each in leads:
        d = df.actual_Y[df.Lead==each]
        d.name = each
        d = d.dropna()
        xtmajor = range(0,len(d)+1,60)  #1/5 of s
        xtminor = range(0,len(d)+1,8)  # 1/25 of a sec
        ytmajor = [x / 100.0 for x in range(-150, 151, 50)]
        ytminor = [x / 100.0 for x in range(-150, 151, 10)]
        d.plot(ax=axes[row,col], use_index=False,ylim=(-2,2), yticks=ytmajor, xticks=xtmajor, grid=True, color='black',legend=True)

        ax = axes[row,col]
        ax.set_xticks(xtmajor)

        ax.set_xticklabels(xtmajor,rotation='vertical')
        ax.set_xticks(xtminor, minor=True)
        ax.set_yticks(ytmajor)
        ax.set_yticks(ytminor, minor=True)

        ax.grid(b=True, which='both',color='r')
        ax.grid(which='minor', alpha=0.1)

        col +=1
        if col + 1 > columns:
            col = 0
            row +=1

    return d

     # This makes the ticks mostly match standard EKG gridlines major blocks (5x5)


df = loadScanCSV()
plotRecordTimeDelta(df)
