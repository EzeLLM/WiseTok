#!/usr/bin/python
#filename:ex5.py
#-*-coding:utf-8 -*-

my_name = 'wolfe_yuan'
my_age = 35
my_height = 72
my_weight = 176
my_eyes = 'Brown'
my_teeth = 'White'
my_hair = "Head"

print "Let's talk about %s." %my_name
print "He's %d inches tall." %my_height
print "He's %d pounds heavy." %my_weight
print "Actually that's not too heavy"
print "He's got %s eyes and %s hair." %(my_eyes, my_hair)
print "His teech and usually %s depending on the coffee and tea." %my_teeth

#this line is tricky, try to get it exactly right
print "If i add %d, %d, and %d I get %d."%(my_age, my_height, my_weight, my_age+my_height+my_weight)