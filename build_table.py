#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:28:54 2022

@author: francesco
"""

from io import StringIO
import pandas as pd
import os

#read files inside the folder
path = 'pearson_tables'
content = os.listdir(path)
content.sort()

#identify and divide tp and t2m
tp = []
t2m = []

for item in content:
    if item.split('_')[0] == 'tp':
        tp.append(path+'/'+item)
    elif item.split('_')[0] == 't2m':
        t2m.append(path+'/'+item)
        
months_dict = {
     '01':'Jan',
     '02':'Feb',
     '03':'Mar',
     '04':'Apr',
     '05':'May',
     '06':'Jun',
     '07':'Jul',
     '08':'Aug',
     '09':'Sept',
     '10':'Oct',
     '11':'Nov',
     '12':'Dec'
    }   

#months_dict = dict(zip([str(i+1) for i in range(12)],months_dict.values()))     

#write tp sheet
writer = pd.ExcelWriter('pearson_tables/summary.xlsx',engine='xlsxwriter')
workbook=writer.book
worksheet=workbook.add_worksheet('tp')
writer.sheets['tp'] = worksheet
#for each file
for i,item in enumerate(tp):
    #give it a name corresponding to the month
    name = months_dict[item.split('/')[-1].split('-')[-1].split('.')[0]]

    #open the file
    table = pd.read_excel(item, index_col=0).astype(float).round(2)
   
    #write it in the excel sheet
    aggr = item.split('/')[-1].split('-')[-2]
    if aggr == '1':
        table.to_excel(writer,sheet_name='tp',startrow=1 , startcol=(i*4))
        worksheet.write_string(0, (4*i+1), name)
        save = i*4
    elif aggr == '2':
        table.to_excel(writer,sheet_name='tp',startrow=12 , startcol=(i*4-save-4))

        save2 = i*4-save
    elif aggr == '3':
        table.to_excel(writer,sheet_name='tp',startrow=24 , startcol=(i*4-save2-save-4))


worksheet=workbook.add_worksheet('t2m')
writer.sheets['t2m'] = worksheet
for i,item in enumerate(t2m):
    #give it a name corresponding to the month
    name = months_dict[item.split('/')[-1].split('-')[-1].split('.')[0]]

    #open the file
    table = pd.read_excel(item, index_col=0).astype(float).round(2)
   
    #write it in the excel sheet
    aggr = item.split('/')[-1].split('-')[-2]
    if aggr == '1':
        table.to_excel(writer,sheet_name='t2m',startrow=1 , startcol=(i*4))
        worksheet.write_string(0, (4*i+1), name)
        save = i*4
    elif aggr == '2':
        table.to_excel(writer,sheet_name='t2m',startrow=12 , startcol=(i*4-save-4))

        save2 = i*4-save
    elif aggr == '3':
        table.to_excel(writer,sheet_name='t2m',startrow=24 , startcol=(i*4-save2-save-4))


#write t2m sheet
writer.save()






'''
writer = pd.ExcelWriter('/Users/francesco/Desktop/newNIPA/pearson_tables/test.xlsx',engine='xlsxwriter')
workbook=writer.book
worksheet=workbook.add_worksheet('tp')
writer.sheets['tp'] = worksheet
worksheet.write_string(0, 0, df1.name)
'''













