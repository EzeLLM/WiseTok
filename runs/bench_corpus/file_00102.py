Python 2.7.6 (default, Jun 22 2015, 18:00:18) 
[GCC 4.8.2] on linux2
Type "copyright", "credits" or "license()" for more information.
>>> x = 5
>>> if x<10:
	print('Smaller')

	
Smaller
>>> if x>20:
	print('Bigger')

	
>>> if x>20:
	print('Bigger')
	print('Finis')

	
>>> 
>>> x=5
>>> if x==5:
	print("Equals 5")

	
Equals 5
>>> if x>4:
	print("Greater than 4")

	
Greater than 4
>>> if x>=5:
	print("Greater than or equals 5")

	
Greater than or equals 5
>>> if x<6:
	print("Less than 6")

	
Less than 6
>>> if x!=6:
	print("Not equal 6")

	
Not equal 6
>>> x=5
>>> print("Before 5")
Before 5
>>> if x==5
SyntaxError: invalid syntax
>>> if x == 5:
	print("Is 5")
	print("Is still 5")
	print("Third 5")

	
Is 5
Is still 5
Third 5
>>> print("Afterwards 5")
Afterwards 5
>>> print("Before 6")
Before 6
>>> if x==6:
	print("Is 6")
	print("Is still 6")
	print("Third 6")

	
>>> print("Afterwards 6")
Afterwards 6
>>> x=5
>>> if x>2:
	print("Bigger than 2")
	print("Still bigger")
print("Done with 2")
SyntaxError: invalid syntax
>>> ================================ RESTART ================================
>>> 
Bigger than 2
Still bigger
Done with 2
>>> for i in range(5):
	print(i)
	if i>2:
		print("Bigger than 2")
	print("Done with i",i)
print("All done")
SyntaxError: invalid syntax
>>> ================================ RESTART ================================
>>> 
0
('Done with i', 0)
1
('Done with i', 1)
2
('Done with i', 2)
3
Bigger than 2
('Done with i', 3)
4
Bigger than 2
('Done with i', 4)
All done
>>> ================================ RESTART ================================
>>> 
More than one
Less than 100
All done
>>> ================================ RESTART ================================
>>> 
Bigger
All done
>>> ================================ RESTART ================================
>>> 

Traceback (most recent call last):
  File "/home/agneta/RTR105/piemers1.py", line 1, in <module>
    if x<2:
NameError: name 'x' is not defined
>>> ================================ RESTART ================================
>>> 
Medium
All done
>>> ================================ RESTART ================================
>>> 
LARGE
All done
>>> 
