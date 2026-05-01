import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit
from optparse import OptionParser, OptionGroup

def getAttribute(cur, planners, attribute, typename):
    labels = []
    measurements = []
    nanCounts = []
    if typename == 'ENUM':
        cur.execute('SELECT description FROM enums where name IS "%s"' % attribute)
        descriptions = [ t[0] for t in cur.fetchall() ]
        numValues = len(descriptions)
    for planner in planners:
        cur.execute('SELECT %s FROM runs WHERE plannerid = %s AND %s IS NOT NULL' \
            % (attribute, planner[0], attribute))
        measurement = [ t[0] for t in cur.fetchall() if t[0] != None ]
        if len(measurement) > 0:
            cur.execute('SELECT count(*) FROM runs WHERE plannerid = %s AND %s IS NULL' \
                % (planner[0], attribute))
            nanCounts.append(cur.fetchone()[0])
            labels.append(planner[1])
            if typename == 'ENUM':
                scale = 100. / len(measurement)
                measurements.append([measurement.count(i)*scale for i in range(numValues)])
            else:
                measurements.append(measurement)

    if len(measurements)==0:
        print('Skipping "%s": no available measurements' % attribute)
        return

    dic = {}
    for label,measurement in zip(labels,measurements):
        dic[label] = measurement
    return dic


def read_database(dbname):
    conn = sqlite3.connect(dbname)
    conn.text_factory = str
    c = conn.cursor()
    c.execute('PRAGMA FOREIGN_KEYS = ON')
    c.execute('SELECT id, name FROM plannerConfigs')
    planners = [(t[0],t[1].replace('geometric_','').replace('control_',''))
        for t in c.fetchall()]
    c.execute('PRAGMA table_info(runs)')
    colInfo = c.fetchall()[3:]
    # print(colInfo)
    data = {}
    for planner in planners:
        # print(planner[1])
        data[planner[1]] = {}
    for col in colInfo:
        if col[2] == 'BOOLEAN' or col[2] == 'ENUM' or \
            col[2] == 'INTEGER' or col[2] == 'REAL':
            attribute = getAttribute(c, planners, col[1], col[2])
            for planner in attribute:
                data[planner][col[1]] = attribute[planner]
    return data


def plot_attribute(data,planners,attribute_name,ax,args,label):
    atts = [data[planner][attribute_name] for planner in planners]
    att_avg = [sum(att)/len(att) for att in atts]
    return ax.plot(att_avg,args,label=label)
    

def plot_rrt_time(data1,data2,data3):
    fig=plt.figure() #Creates a new figure
    ax1=fig.add_subplot(111) #Plot with: 1 row, 1 column, first subplot
    planners = [planner for planner in data1]
    planners.sort()
    plot_attribute(data1,planners,"time",ax1,'ko-','Denso')
    plot_attribute(data2,planners,"time",ax1,'bo-','Estreito')
    plot_attribute(data3,planners,"time",ax1,'ro-','Cercado')
    axes = plt.gca()
    axes.set_xlim([-0.2,len(planners)-1+0.2])
    plt.xticks(range(len(planners)), planners, size='small')
    plt.legend()
    plt.show()

def plot_3_env(data1,data2,data3,att_name,ylabel):
    fig=plt.figure() #Creates a new figure
    ax1=fig.add_subplot(111) #Plot with: 1 row, 1 column, first subplot
    planners = [planner for planner in data1]
    planners.sort()
    plot_attribute(data1,planners,att_name,ax1,'ko-','Denso')
    plot_attribute(data2,planners,att_name,ax1,'bo-','Estreito')
    plot_attribute(data3,planners,att_name,ax1,'ro-','Cercado')
    axes = plt.gca()
    axes.set_xlim([-0.2,len(planners)-1+0.2])
    plt.xticks(range(len(planners)), planners, size='small')
    plt.legend()
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    data_denso = read_database("databaseRRTPassoDenso.db")
    data_estreito = read_database("databaseRRTPassoEstreito.db")
    data_cercado = read_database("databaseRRTPassoCercado.db")

    # print(data1.values()[0].keys())

    plot_3_env(data_denso,data_estreito,data_cercado,"time",'Tempo (s)')
    plot_3_env(data_denso,data_estreito,data_cercado,"simplified_solution_length",'Comprimento do caminho')
    