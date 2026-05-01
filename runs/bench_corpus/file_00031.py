# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:41:06 2016

@author: pqian
"""
import sys
import os
import win32com.client as win32
import pyodbc
import re
import getpass
import pandas
from datetime import datetime


class automate_reports:
    
    ##### Initialize some variables in the class
    def __init__(self, viewSet, userID, password):
        self.viewSet = viewSet
        self.viewSet_ytd = 'YTD' + self.viewSet
        self.viewSet_rol = 'ROL' + self.viewSet
        self.currMth= self.viewSet[-6:]
        self.mthTimeKey = self.currMth + '01'
        self.theFolder = os.getcwd()
        self.rootFolder = self.theFolder + '\\Internal Reporting'
        self.folder_ex = self.theFolder + '\\External Reporting'
        self.PO_app = self.folder_ex + '\\PO App.xlsm' # indicate the paths for Excel apps; REMEMBER, must provide the full path (this is for VBA not for Python)
        self.sqlPath = 'Internal Reporting\\Queries\\SQL Queries_wprod.sql' # Path for all sql queres
        self.sqlPaht_drop = 'Internal Reporting\\Queries\\SQL Drop Tables_wprod.sql'
        self.newfolder_ext = self.folder_ex + '\\Cargo\\' + self.viewSet
        self.newFolder = self.rootFolder + '\\Cargo\\' + self.viewSet
        self.docFolder = self.rootFolder + '\\Docs\\'
        self.excel = win32.gencache.EnsureDispatch('Excel.Application') # Set up excel application connection
        self.adodb = win32.gencache.EnsureDispatch('ADODB.Connection') # Set up ADODB Connection for database connection
        self.rs = win32.gencache.EnsureDispatch('ADODB.Recordset') # Set up ADODB Recordset to get query results
        self.xlConst = win32.constants # constants like "xl..." in VBA
        # Login information for FCTSREPP database connection
        self.uid = userID
        self.psw = password
        self.DSN = 'DRIVER={Oracle in OraClient11g_home1};SERVER=WPROD.HAP.ORG;DBQ=WPROD.HAP.ORG;UID=' +self.uid+';PWD='+self.psw
        self.dbConn = pyodbc.connect(self.DSN, autocommit=True)
        self.dbExe = self.dbConn.cursor()
        self.FMT = '%H:%M:%S' # for time formatting
        
    @staticmethod
    def get_month_name(date): # month sould be in a format of 'yyyymmdd'
        mon = datetime.strptime(date, '%Y%m')
        month = mon.strftime('%b')
        return month
        
    ##### Run all sql queries for tables update   
    #@staticmethod
    def execute_sql_file(self, inFile, dbConn):
        #######################################
        # Executes all sql in a file for a
        # pyodbc connection
        #######################################
        #Read file as string

        with open(inFile) as dat:
            sql = dat.read() # read file
        sqlStr = sql.replace('currReviewSet_rol', self.viewSet_rol).replace('currReviewSet', self.viewSet_ytd).replace('currMth', self.currMth).replace('mthTimeKey', self.mthTimeKey)
        #Pattern matches -- and /**/ style comments and any newlines, tabs or multiple spaces
        pattern = re.compile('(--[^(\r|\n)]*)|(\/\*[^\*]*\*\/)|(\s+)')
        rm_pattern = ' '.join(re.sub(pattern, " ", sqlStr).split())
        
        # Execute the command of drop tables
        # Split on semicolon to get each statement; remove null statement following final semicolon
        statements = [s for s in re.split('; |;', rm_pattern) if s != ""]
        
        for s in statements:
            try:
                dbConn.execute(s)
            except Exception as e:
                print (s)
                print(repr(e))
                sys.exit()
            
    def update_database(self, inFile, antFile, dbConn):
        # drop all the
        with open(antFile) as F:
            sqlString = F.read()
        dbConn.execute(sqlString)
        # update database(create/recreate tables)
        print ('Updating database by recreating tables')
        self.execute_sql_file(inFile, dbConn)
        print ('Daatabase Updated')
        
    ##### Run excel application for PO reports
    def run_excel_app(self, appPath):
        print ('Creating PO Reports')
        wb = self.excel.Workbooks.Open(appPath) # Open the application in Excel
        wb.Visible = False
        wsh = wb.Worksheets('Main')
        wsh.Range('A2').Value = self.viewSet_ytd # Change the view set to the current report time period
        wsh.Range('G2').Value = self.uid # get the user name for database connection in Excel app
        wsh.Range('G3').Value = self.psw # get the password for database connection in Excel app
        self.excel.Application.Run('Main') # start executing VBA code to run reports
        wsh.Range('G2').Value = " "
        wsh.Range('G2').Value = " "
        wb.Close(False) # close the Excel
        print ('All PO Reports are already created!')
        
    ##### Open ADODB using wincom32
    def open_adodb(self, sqlPath, wbPath, savePath, sheetList):
        self.adodb.Open(self.DSN) # open adodb for database connection
        wb = self.excel.Workbooks.Open(wbPath)
        wb.Visible = True
        wb.SaveAs(savePath) 
        for i in range(0, len(sheetList)): # loop through sheets in sheet list for data loading
            wsh = wb.Sheets(sheetList[i])
            print (wsh.Name)
            wsh.ListObjects(1).Unlist # clear table format
            wsh.Cells.Clear # clear all cells in the sheet
            with open(sqlPath[i], 'r') as file: # read sql in text file as string
                sqlStr = file.read()
            self.rs.Open(sqlStr, self.adodb) # open recordset and run the query in it
            for j in range(0, self.rs.Fields.Count-1): # get the field names
                wsh.Range('A1').GetOffset(0, j).Value = self.rs.Fields(j).Name
            wsh.Range('A2').CopyFromRecordset(self.rs) # copy all recordset(query results) to the cells
            tblRange = wsh.UsedRange
            listObj = wsh.ListObjects.Add(self.xlConst.xlSrcRange, tblRange, False, self.xlConst.xlYes)
            listObj.TableStyle = 'TableStyleMedium2'
            listObj.Name = wsh.Name.replace(' - ', '')
            for e in listObj.HeaderRowRange: # get names of headers
                if e.Value.lower().find('date') != -1: # format the columns of "DATE"
                    listObj.ListColumns(e.Value).DataBodyRange.NumberFormat = 'MM/DD/YYYY'
        wb.Close(True)
        
        
    ##### This function is used to add table style and format the table with formalizing date type
    def add_table_style(self, savePath, sheetList):
        wb = self.excel.Workbooks.Open(savePath)
        for wsh in wb.Worksheets(sheetList):
            print ('Formatting Sheet %s' %wsh.Name)
            wsh.Columns(1).EntireColumn.Delete() # delete the first column
            tblRange = wsh.UsedRange # catch all the used cells
            listObj = wsh.ListObjects.Add(self.xlConst.xlSrcRange, tblRange, False, self.xlConst.xlYes) # add a new table
            listObj.TableStyle = 'TableStyleMedium2' # add a table style to the table
            listObj.Name = wsh.Name.replace(' - ', '') # format table name
            for e in listObj.HeaderRowRange: # get names of headers
                if e.Value.lower().find('date') != -1: # format the columns of "DATE"
                    try:
                        listObj.ListColumns(e.Value).DataBodyRange.NumberFormat = 'MM/DD/YYYY'
                    except Exception as e:
                        print (repr(e))
        wb.Close(True)

        
    ##### Run sql query and fetch data into pandas
    def oracle_to_pandas(self, inFile):
        print ('Pulling out data from Oracle using sql %s' %inFile)
        sqlFile = open(inFile)
        sqlStr = sqlFile.read()
        dataOra = pandas.read_sql(sqlStr, self.dbConn) # Using pandas to read data from oracle exporting into a Python string variable
        df = pandas.DataFrame(dataOra)
        return df
    
    ##### Create a workbook and save
    def create_save_wb(self, savePath, sheetList):
        wb = self.excel.Workbooks.Add()
#        self.excel.Application.DisplayAlerts = False # so excel files can be overwritten without showing alert messages
        wb.SaveAs(savePath)
#        self.excel.Application.DisplayAlerts = True
        for sheet in sheetList:
            print ('Adding Worksheet %s' %sheet)
            wb.Worksheets.Add().Name = sheet
            
        wb.Close(True)
        
    ##### Write data in Pandas to Excel
    def pandas_to_excel(self, savePath, sqlList, sheetList):
        writer = pandas.ExcelWriter(savePath, engine='xlsxwriter')
        for i in range(0, len(sheetList)): # loop through sheets in sheet list for data loading
            print ('Populating sheet %s' %sheetList[i])
            df = self.oracle_to_pandas(sqlList[i]) # run sql query in oracle for data extraction
            df.to_excel(writer, sheet_name = sheetList[i]) # load data from pandas data frame to one Excel sheet
            
        writer.save() 
        
    ##### This function contains three functions: create workbook, load data to excel and add table format
    def wb_toexcel_format(self, savePath, sheetList, sqlList):
        self.create_save_wb(savePath, sheetList)
        self.pandas_to_excel(savePath, sqlList, sheetList)
        self.add_table_style(savePath, sheetList)
        
    ##### CBHM Behavioral Health Rates
    def hedis_rpt_cbhm_bh(self, filePath):
        folderName = '\\HEDIS_RPT_CBHM_BEHAVIORALHEALTH_' + self.viewSet + '.xlsx'
        savePath = self.newFolder + folderName
        new_wsh = 'Report - Rates'
        sheetList = ['Report - Gaps', 'Data - Rates']
        sqlList = ['Internal Reporting\\Queries\\hedis_rpt_cbhm_gaps.sql', 'Internal Reporting\\Queries\\hedis_rpt_cbhm_rates.sql']
        self.wb_toexcel_format(savePath, sheetList, sqlList)
        ### Refresh pivot's data source
        print ('Refreshing pivot table data source')
        wb = self.excel.Workbooks.Open(savePath)
        wb_tem = self.excel.Workbooks.Open(filePath)
        wb_tem.Sheets(new_wsh).Copy(Before=wb.Sheets(sheetList[0])) # copy sheet from the template workbook to the current active workbook
        #return None
        wsh = wb.Sheets(new_wsh)
        wsh.PivotTables(1).ChangePivotCache(wb.PivotCaches().Create(self.xlConst.xlDatabase, SourceData = 'DataRates')) # update the data source of pivot table                                                                                                                      
        wsh.Range("A2").Formula = '="HEDIS Review Set: " & INDEX(DataRates[HEDIS_REVIEW_SET],1,1)'
        wb.Close(True)
        wb_tem.Close(False)
        print ('CBHM Behavioral Health Rates is done')
        
    ##### CBHM FUMIL Discharges and Gaps
    def hedis_rpt_fumil_discharge_gaps(self):
        folderName = '\\HEDIS_RPT_CBHM_FUMIL_DISCHARGE_GAPS_' + self.viewSet + '.xlsx'
        savePath = self.newFolder + folderName
        sheetList = ['Discharge', 'Gaps']
        sqlList = ['Internal Reporting\\Queries\\hedis_rpt_fumil_dc.sql', 'Internal Reporting\\Queries\\hedis_rpt_fumil_gaps.sql']
        self.wb_toexcel_format(savePath, sheetList, sqlList)
        print ('CBHM FUMIL Discharges and Gaps is done')
        
    ##### CMDM Gaps in Care
    def hedis_rpt_cmdm_gaps(self, filePath):
        folderName = '\\HEDIS_RPT_CMDM_GAPSINCARE_' + self.viewSet + '.xlsx'
        savePath = self.newFolder + folderName
        new_wsh = 'Criteria'
        sheetList = ['Data - VEBA', 'Data - MA, Non-VEBA', 'Data - Commercial, Non-VEBA']
        sqlList = ['Internal Reporting\\Queries\\hedis_rpt_cmdm_gaps_VEBA.sql', 'Internal Reporting\\Queries\\hedis_rpt_cmdm_gaps_nonVEBA_ma.sql', 'Internal Reporting\\Queries\\hedis_rpt_cmdm_gaps_nonVEBA_comm.sql']
        self.wb_toexcel_format(savePath, sheetList, sqlList)
        ### Copy a criteria sheet from template to current workbook
        wb = self.excel.Workbooks.Open(savePath)
        wb_tem = self.excel.Workbooks.Open(filePath)
        wb_tem.Sheets(new_wsh).Copy(Before=wb.Sheets(sheetList[0]))
        if self.viewSet[-2:] == '12':
            mon_num = '201601'
        else:
            mon_num = str(int(self.viewSet)+1)
        wb.Sheets(new_wsh).Range('B3').Value = '%s %s - %s %s' %(self.get_month_name(mon_num), str(int(self.viewSet[:4])-1), self.get_month_name(self.viewSet), self.viewSet[:4])
        wb.Close(True)
        wb_tem.Close(False)
        print ('CMDM Gaps in Care is done')
        
    ##### HEDIS 5 Star Metrics
    def hedis_rpt_5star_metrcs(self, sFileName):
        fileName = '\\Mark - 5 Star Metric_%s.xlsx' %self.viewSet
        savePath = self.newFolder + fileName
        sqlPath = 'Internal Reporting\\Queries\\hedis_rpt_5star_metrics.sql'
        wbPath = self.rootFolder + sFileName
        sheetList = ['Data']
        self.open_adodb(sqlPath, wbPath, savePath, sheetList)
        wb = self.excel.Workbooks.Open(wbPath)
        wsh = wb.Sheets('Report')
        wsh.PivotTables(1).Refresh()
        wb.Close(True)
 
    ##### HEDIS Immunization Mailing
    def hedis_rpt_immunization_mailing(self):
        folderName = '\\HEDIS_RPT_IMMUNIZATION_MAILING_' + self.viewSet + '.xlsx'
        savePath = self.newFolder + folderName
        sheetList = ['Immunization Mailing']
        sqlList = ['Internal Reporting\\Queries\\hedis_rpt_immunization_mailing.sql']
#        openPath = self.docFolder + 'HEDIS_Immnunization Mailing.xlsx'
        self.wb_toexcel_format(savePath, sheetList, sqlList)
#        self.open_adodb(sqlList, openPath, savePath, sheetList)
        
    ##### MIHIN Member Extract
    def hedis_rpt_mihin(self):
        folderName = '\\HAP_ACRSATTRIBUTION_' + self.viewSet + '.xlsx'
        savePath = self.newFolder + folderName
        sheetList = ['ASRS Attribution']
        sqlList = ['Internal Reporting\\Queries\\hedis_rpt_mihin.sql']
        self.wb_toexcel_format(savePath, sheetList, sqlList)
        
    ##### Start the whole process
    def start(self):
        stTime = datetime.now().strftime(self.FMT)
        print ('Program started at %s' %stTime)
        # create new folder for this reporting period
        if not os.path.exists(self.newFolder):
            os.makedirs(self.newFolder)
        # update tables in Oracle
#        self.update_database(self.sqlPath, self.sqlPaht_drop, self.dbExe)
        # run PO reports
#        self.run_excel_app(self.PO_app)
        # CBHM Behavioral Health Rates
#        self.hedis_rpt_cbhm_bh(self.rootFolder + '\\Docs\\HEDIS CBHM Rates.xlsx')
        # CBHM FUMIL Discharges and Gaps
#        self.hedis_rpt_fumil_discharge_gaps()
        # CMDM Gaps in Care
        self.hedis_rpt_cmdm_gaps(self.rootFolder + '\\Docs\\HEDIS CMDM Gaps In Care.xlsb')
        # 5 Star Metrics
#        self.hedis_rpt_5star_metrcs(self, 'Docs\\Mark - 5 Star Metrics.xlsx')

            #       #  HEDIS Immunization Mailing
            #        self.hedis_rpt_immunization_mailing()

        # TODO(Peng): MATRIX In Home Screening Gaps

        # MIHIN Member Extract
#        self.hedis_rpt_mihin()
        # Need to quit excel application
        self.excel.Application.Quit()
        endTime = datetime.now().strftime(self.FMT)
        costTime = datetime.strptime(endTime, self.FMT) - datetime.strptime(stTime, self.FMT)
        print ('Program is completed, at %s, costing %s' %(endTime, costTime))

if __name__ == '__main__':
    viewSet = '201612' # Define the report time period
    userID = input('WPROD User ID:') # input database user ID
    password = getpass.getpass('WPROD Password:') # input database password
    run_program = automate_reports(viewSet, userID, password) # inherit the class "automate_reprots"
    run_program.start() # execute a function in automate_reports to start the task
