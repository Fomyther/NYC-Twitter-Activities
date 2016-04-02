#!/usr/bin/env python
#-*- coding: utf-8 -*-

__author__ = "Stephen"
__date__ = "2016_03_29"

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import os
import operator
from datetime import datetime, timedelta, date
import time
import geopandas as gp

PWD = '/Users/CQ/Documents/Project1'
zipPath = PWD + '/nyc-zip-code-tabulation-areas-polygons.geojson'
geo_NY = gp.read_file(zipPath)[['geometry', 'postalCode']]
twitter_file = pd.read_csv(PWD + '/2016_02_18_sintetic.csv')
USzipcode = pd.read_csv(PWD + '/USzipcode_XY.csv')

LABEL = ['All ','Week','Satur','Sun']

def Clean_zipcode(dataset):
    zipcode_all = list(sorted(set(dataset.ZipCode)))
    zip_exist = [i for i in zipcode_all if i in USzipcode.ZipCode.values]
    return dataset[dataset.ZipCode.isin(zip_exist)]

TimeList = range(4)
TimeList[0] = [time.localtime(x) for x in twitter_file.timestamp]
TimeList[1] = [x for x in TimeList[0] if x.tm_wday < 5]
TimeList[2] = [x for x in TimeList[0] if x.tm_wday == 5]
TimeList[3] = [x for x in TimeList[0] if x.tm_wday == 6]
twitter_file.iloc[:,0] = TimeList[0]

twitter_file.columns = ['DateTime','Stamp','ZipCode','User','ID']
USzipcode.columns = ['ZipCode','lat','lon']

Data = range(4)
for i in range(4):
    Data[i] = twitter_file[twitter_file.DateTime.isin(TimeList[i])]
    Data[i] = Clean_zipcode(Data[i])

PLACES = Data[0].ZipCode.unique()

def Graph_build(network, date_index, directed = True):
    '''This function is for creating graphs based on the networks we have.
    The argument 'directed' indicates whether this graph is directed,
    default setting is True
    '''
    Graph = nx.DiGraph() if directed == True else nx.Graph()
    places = Data[date_index].ZipCode.unique()
    Graph.add_nodes_from(places)
    
    for i in range(len(network[date_index])):
        w = network[date_index].iloc[i,:]
        Graph.add_edge(w[0],w[1],weight = w[2])
    return Graph

#add unit weights to unweighted network nodes
def make_directed_and_weighted(G):
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    
    for e in G.edges():
        v1 = min(e[0],e[1])
        v2 = max(e[0],e[1])
        w = G[v1][v2]['weight']
        #if it's a loop edge, we double its weight
        if v1==v2:
            DG.add_edge(v1,v2, weight = 2 * w)
        #otherwise we set the weight of A->B equal to the weight of B->A
        else:
            DG.add_edge(v1,v2, weight = w)
            DG.add_edge(v2,v1, weight = w)
    return DG

#produce symmetrized undirected version of a directed network
def as_undirected(G):
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    for e in G.edges():
        UG.add_edge(e[0],e[1], weight = G[e[0]][e[1]]['weight'])
    return UG

def TopZipCode(fig, rank, date_index, Color, top = 10):    
    ziplist = sorted(rank.items(), key=operator.itemgetter(1), reverse = 1)
    x = [i[0] for i in ziplist[:top]]
    y = [i[1] for i in ziplist[:top]]
    ax = fig.add_subplot(221 + date_index)
    plt.title(LABEL[date_index]+'days', fontweight="bold", size=15)
    ax.bar(range(top), y, color = Color)
    ax.set_xticks(range(top))
    ax.set_xticklabels(x, fontsize = 13-top/4, fontweight="bold", rotation = 30)
    plt.subplots_adjust(hspace = 0.2, wspace = 0.1)
    return x


def PartitionSorting(partition):
    ClusterList = [ (j, partition.values().count(j)*1.0/len(partition)) for j in set(partition.values())]
    ClusterList.sort(key = operator.itemgetter(1), reverse = 1)
    ClusterRanking = [i[0] for i in ClusterList]
    sorted_partition = {i:0 for i in partition.keys()}
    for i in sorted_partition.keys():
        sorted_partition[i] = ClusterRanking.index(partition[i]) + 1
    return sorted_partition


def PlotMapPart(fig, partition, date_index=0, zips=geo_NY, key='postalCode', cmap='spectral', title='Community Detection',size=221, alpha=.7):
    ax = fig.add_subplot(size + date_index)
    y = {i:'Community %d%d'%(partition[i]/10,partition[i]%10) for i in partition.keys()}
    p = pd.Series(y).reset_index().rename(columns={'index':'postalCode',0:'part'})
    zips.postalCode = zips.postalCode.astype(int)
    z = zips.merge(p, on=key, how='left')
    level = len(set(partition.values()))
    z[z.part.notnull()].plot(column = 'part', categorical=1, ax=ax, alpha=alpha, cmap = cmap, legend=True)
    plt.title(title, fontweight="bold", size=15)
    ax.axis('off')
    plt.subplots_adjust(hspace = 0.1, wspace = 0)


def PlotPageRankPart(fig, partition, TopZip, date_index=0, zips=geo_NY, key='postalCode', cmap='YlOrRd', title='Community Detection',size=221, alpha=.7):
    ax = fig.add_subplot(size + date_index)
    p = pd.Series(partition).reset_index().rename(columns={'index':'postalCode',0:'part'})
    zips.postalCode = zips.postalCode.astype(int)
    z = zips.merge(p, on=key, how='left')
    z[z.part.notnull()].plot(column = 'part', ax=ax, alpha=alpha, cmap = cmap, categorical=1, legend = 0)
    z[z.postalCode.isin(TopZip[date_index])].plot(ax=ax, alpha=alpha, color = 'k', categorical=1, legend = 0)
    plt.title(title, fontweight="bold", size=15)
    ax.axis('off')
    plt.subplots_adjust(hspace = 0.1, wspace = 0)


def DailyActivity():
    start_date = datetime.fromtimestamp(time.mktime(TimeList[0][0])).date()
    end_date = datetime.fromtimestamp(time.mktime(TimeList[0][-1])).date()
    day_count = (end_date-start_date).days
    DateList = [start_date + timedelta(n) for n in range(day_count)]
    T_ACT = {i:0 for i in DateList}
    for single_day in DateList:
        day_list = [x for x in Data[0].DateTime if datetime.fromtimestamp(time.mktime(x)).date() == single_day]
        T_ACT[single_day] = len(day_list)
    Thanksgiving = date(2015, 11, 26)
    Christmas = date(2015, 12, 25)
    NewYear = date(2016, 1, 1)
    a = Thanksgiving - start_date
    b = date(2015,12,1) - start_date
    c = Christmas - start_date
    d = NewYear - start_date
    e = date(2016,2,1) - start_date
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111)
    ax.set_title("Daily Twitter Activities", fontweight="bold", size=18)
    bar = ax.bar(range(day_count), T_ACT.values(), color = ['g','g','g','r','r','g','g'] , align='center', width = 1.5)
    ax.legend((bar[0],bar[4]), ('Weekdays', 'Weekends'), loc = 2, prop={'size':18})
    ax.set_xticks([0,a.days, b.days, c.days, d.days, e.days])
    ax.set_xticklabels(['Nov/4/2015','Thanksgiving','Dec/1/2015','Christmas','New Year','Feb/1/2016'],
                       rotation=90, fontsize = 14)
    plt.show()


def ShowFeature(demo_city, feature_name, sorted_partition, LEGEND = None, title = 'Community Data', date_index = 0):
    '''date_index has the similar function here: controlling the sample base
    0--All days
    1--Weekdays
    2--Saturdays
    3--Sundays
    with 0 being the default'''
    #please make sure to use sorted_partition so that ClusterRanking is no longer needed
    partition = sorted_partition[date_index]
    feature_len = len(feature_name)
    data = demo_city.loc[:,feature_name]
    data.fillna(value = 0, inplace=True)
    data = data.applymap(int)
    data['ZipCode'] = demo_city.zipcode
    data = data[data.ZipCode.isin(set(partition.keys()))]
    data['Part'] = [partition[i] for i in data.ZipCode]
    if feature_len > 1:
        ix_feature = ' percentage '
        SUM = data.iloc[:,0:feature_len].sum(axis = 1)
        ix_sum = SUM!=0
        data = data[ix_sum]
        for i in range(feature_len):
            data.iloc[:,i] /= SUM[ix_sum]
    else:
        ix_feature = ' data '
    grouped = data.iloc[:,:-2].groupby(data.Part)
    STD = grouped.std()
    MEAN = grouped.mean()
    level = len(set(partition.values()))
    Width = 3.0/feature_len/level
    fig = plt.figure(figsize = (20, 9))
    x = range(level)
    LABEL = ['Community %d%d'%((i+1)/10,(i+1)%10) for i in x] 
    bar_step = np.arange(0, feature_len)
    Color = matplotlib.cm.spectral(np.linspace(0,1,feature_len+1))
    ax = fig.add_subplot(111)
    for i in range(feature_len):
        plt.bar(x+bar_step[i]*Width, MEAN.iloc[:,i] ,width = Width, yerr = list(STD.iloc[:,i]), color = Color[i+1], align='center')
    Title = ['ALL days','Weekdays', 'Saturdays', 'Sundays']
    plt.title(title+ix_feature+'on each community (Partition based on '+Title[date_index]+')', fontweight="bold", size = 18)
    y = [i+(feature_len-1)*0.5*Width for i in x]
    ax.set_xticks(y)
    ax.set_xticklabels(LABEL, fontsize = 14, fontweight="bold")
    if LEGEND != None:
        plt.legend(LEGEND, fontsize = 18-feature_len/2, loc =2)
    plt.grid()
    plt.show()


def ShowFeatureWBP(demo_city, feature_name, sorted_partition, LEGEND = None, title = 'Community Data', date_index = 0):
    '''date_index has the similar function here: controlling the sample base
    0--All days
    1--Weekdays
    2--Saturdays
    3--Sundays
    with 0 being the default
    WBP stands for "weighted by population"
    '''
    #please make sure to use sorted_partition so that ClusterRanking is no longer needed
    partition = sorted_partition[date_index]
    feature_len = len(feature_name)
    data = demo_city.loc[:,['zipcode','SE_T001_001']+feature_name]
    data.fillna(value = 0, inplace=True)
    data = data.applymap(int)
    data = data[data.zipcode.isin(set(partition.keys()))]
    data['Part'] = [partition[i] for i in data.zipcode]
    if feature_len > 1:
        ix_feature = ' percentage '
        SUM = data.iloc[:,2:-1].sum(axis = 1)
        ix_sum = SUM!=0
        data = data[ix_sum]
        for i in range(feature_len):
            data.iloc[:,i+2] /= SUM[ix_sum]
    else:
        ix_feature = ' data '
    part_list = list( set( partition.values() ) )
    level = len(part_list)
    weighted_mean, weighted_std = range(level), range(level)
    for i in part_list:
        temp = data[ data.Part == i ]
        total_pop = temp['SE_T001_001'].sum()
        weighted_mean[part_list.index(i)] = np.dot(temp['SE_T001_001'], temp.iloc[:,2:-1])/total_pop
        c = temp.iloc[:,2:-1] - weighted_mean[part_list.index(i)]
        weighted_std[part_list.index(i)] = np.sqrt( np.dot(temp['SE_T001_001'], c**2) /total_pop )

    MEAN = pd.DataFrame(weighted_mean)
    STD = pd.DataFrame(weighted_std)
    Width = 3.0/feature_len/level
    fig = plt.figure(figsize = (20, 9))
    x = range(level)
    LABEL = ['Community %d%d'%((i+1)/10,(i+1)%10) for i in x] 
    bar_step = np.arange(0, feature_len)
    Color = matplotlib.cm.spectral(np.linspace(0,1,feature_len+1))
    ax = fig.add_subplot(111)
    for i in range(feature_len):
        plt.bar(x+bar_step[i]*Width, MEAN.iloc[:,i] ,width = Width, yerr = list(STD.iloc[:,i]), color = Color[i+1], align='center')
    Title = ['ALL days','Weekdays', 'Saturdays', 'Sundays']
    plt.title(title+ix_feature+'on each community (Partition based on '+Title[date_index]+')', fontweight="bold", size = 18)
    y = [i+(feature_len-1)*0.5*Width for i in x]
    ax.set_xticks(y)
    ax.set_xticklabels(LABEL, fontsize = 14, fontweight="bold")
    if LEGEND != None:
        plt.legend(LEGEND, fontsize = 18-feature_len/2, loc =2)
    plt.grid()
    plt.show()
