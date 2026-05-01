# -*- coding: utf-8 -*-

""" Data Visualization HS 17
    Implemenation of Exercise 1
    
    Moritz Eck - 14 715 296"""
    
import pandas as pd
from bkcharts import Donut
from bokeh.models import Title

from bokeh.models import HoverTool
from bokeh.plotting import figure, ColumnDataSource

from numpy import pi

class Figure4:
    """ This class creates the fourth figure (wedge) of exercise 1 """
    
    def __init__(self, data):
        self.data = data
        self.colors = ["black", "blue", "green", "orange", "yellow"]
        
        self.plot = self.createPlot()
        self.plot = self.createPie()
        self.plot = self.createWedges()
        
    def createPlot(self):
        """ Creates the basic layout of the plot and prepares the data """
        data = self.data
        
        # Only use year 2015
        data = data.loc[data['Jahr'] == 2015]
        
        # Only use the quartier Oerlikon 
        data = data.loc[data['Quartier'] == 'Oerlikon']
        
        # Removing the unnecessary columns
        data = data.drop("SexCd", axis=1)
        data = data.drop("StatZoneSort", axis=1)
        data = data.drop("QuarSort", axis=1)
        data = data.drop("Jahr", axis=1)
        data = data.drop("Quartier", axis=1)
        
        # Sorting the Quartier: Oerlikon according to the top five most births in 2015
        data = data.groupby(['StadtZone'])['Births'].sum().reset_index()
        data = data.sort_values('Births', ascending=False)
        data = data.head(5).reset_index(drop=True)
        self.data = data
        
        plot = figure(plot_width=550, plot_height=475)
        
        #Creating the hover tooltip
        hover = HoverTool(tooltips=[("Statistical Zone", "@Zones"), ("Births", "@Births")], 
                          names=['pie', 'wedges'])
        plot.tools.append(hover)
                             
        # Designing the plot (legend, axis alignment & spacing)
        plot.y_range.start = 0
        plot.y_range.end = 6
        plot.x_range.start = 0
        plot.x_range.end = 6
                        
        # X-Axis design
        plot.xgrid.grid_line_color = None
        plot.xaxis.visible = False
                        
        # Y-Axis design 
        plot.yaxis.visible = False
                
        # Legend 
        plot.title = Title(text="The Statistical Zones of Zürich Oerlikon with the Top 5 Number of Births in 2015") 
           
        return plot
    
    def createPie(self):
        """ Creates the pie chart of the top 5 statistical zones with 
            the most births in Zürich Oerlikon in 2015 """
            
        data = self.data
        plot = self.plot
        
        # Total number of Births in Oerlikon in 2015
        sum_births = data['Births'].values.sum()       
        
        # define the percentages for the full circle
        percents = [0]
        percents.append((data['Births'][0] / sum_births) + percents[0])
        percents.append((data['Births'][1] / sum_births) + percents[1])
        percents.append((data['Births'][2] / sum_births) + percents[2])
        percents.append((data['Births'][3] / sum_births) + percents[3])
        percents.append((data['Births'][4] / sum_births) + percents[4])        
        
        # define starts/ends for wedges from percentages of a circle
        starts = [p*2*pi for p in percents[:-1]]
        ends = [p*2*pi for p in percents[1:]]
        
        df = pd.DataFrame({'Zones' : data['StadtZone'].values, 
                           'Births': data['Births'].values}, copy=True)
        
        source = ColumnDataSource(data=df)        
        
        plot.wedge(x=2, y=2, radius=1.5, 
                   start_angle=starts, end_angle=ends, 
                   color=self.colors, name='pie', source=source)     
        
        return plot
    
    def createWedges(self):
        """ Creates a wedge according to its ratio (number of births in relation 
            to the total number) for each of the top 5 statistical zones 
            of Zurich Oerlikon in 2015 """
            
        data = self.data
        plot = self.plot
        
        # Total number of Births in Oerlikon in 2015
        sum_births = data['Births'].values.sum()  
        
        # wedge sizes
        wedge_sizes = []
        
        for value in data['Births'].values:
            wedge_sizes.append((value/sum_births)*2*pi)
                    
        df = pd.DataFrame({'Zones' : data['StadtZone'].values, 
                           'Births': data['Births'].values}, copy=True)
        
        source = ColumnDataSource(data=df)
        
        plot.wedge(x=4.5, y=[5,4,3,2,1], radius=0.5, 
                   start_angle=0, end_angle=wedge_sizes, 
                   color=self.colors, name='wedges', source=source,
                   legend='Zones')
        
        plot.legend.location = "top_left"
        
        return plot