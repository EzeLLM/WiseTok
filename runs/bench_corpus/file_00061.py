#!/usr/bin/env python
# modified from http://elinux.org/RPi_Email_IP_On_Boot_Debian

## --------------------------------------------------
## Author: SIRIO MAQUEA ALVES
## GitHub: @siriomaquea
## Linkedin: https://www.linkedin.com/in/siriomaqueaalves/
## -------------------------------------------------- 

import subprocess
import smtplib
import socket
from email.mime.text import MIMEText
import datetime
import urllib2
import sys, getopt

def main(argv):
   varfrom = ''
   varto = ''
   varsubject = ''
   varmessage = ''
   varsmtp = ''
   varsmtpport = ''
   varpassword = ''
   try:
      opts, args = getopt.getopt(sys.argv[1:],'hf:t:u:o:s:d:p:')
   except getopt.GetoptError:
      print 'execute : python sendemail.py -h to help!'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'python sendemail.py -f <email_from> -t <email_to> -u <subject> -o <message> -s <smtp> -d <smtp_port> -p <password>'
         sys.exit()
      elif opt in ("-f"):
         varfrom = arg
      elif opt in ("-t"):
         varto = arg
      elif opt in ("-u"):
         varsubject = arg.replace('.',' ')
      elif opt in ("-o"):
         varmessage = arg
      elif opt in ("-s"):
         varsmtp = arg
      elif opt in ("-d"):
         varsmtpport = arg
      elif opt in ("-p"):
         varpassword = arg
#   print '-f', varfrom
#   print '-t', varto
#   print '-u', varsubject
#   print '-o', varmessage
#   print '-s', varsmtp
#   print '-d', varsmtpport
#   print '-p', varpassword

   messagefinal = ''
   if "file:" in varmessage:
	arquivomessage = open(varmessage.replace("file:", ""), 'r')
	messagefinal = arquivomessage.read()
	arquivomessage.close()

	
# Change to your own account information
   to = varto
   gmail_user = varfrom
   gmail_password = varpassword
   smtpserver = smtplib.SMTP(varsmtp, varsmtpport)
   smtpserver.ehlo()
   smtpserver.starttls()
   smtpserver.ehlo
   smtpserver.login(gmail_user, gmail_password)
   today = datetime.date.today()
   # Very Linux Specific
   arg='ip route list'
   p=subprocess.Popen(arg,shell=True,stdout=subprocess.PIPE)
   data = p.communicate()
   split_data = data[0].split()
   ipaddr = split_data[split_data.index('src')+1]
   #extipaddr = urllib2.urlopen("http://icanhazip.com").read()
   #my_ip = 'Local address: %s\nExternal address: %s' %  (ipaddr, extipaddr)
   msg = MIMEText(messagefinal)   #, 'html', 'utf-8')
   msg['Subject'] = varsubject +' ( '+ today.strftime('%b %d %Y') + ' )'
   msg['From'] = gmail_user
   msg['To'] = to
   smtpserver.sendmail(gmail_user, [to], msg.as_string())
   smtpserver.quit()


if __name__ == "__main__":
   main(sys.argv[1:])