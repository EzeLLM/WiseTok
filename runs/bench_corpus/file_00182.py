#Matthew Siegel
#5/14/18
#HW6.py 

file = open('engmix.txt')

#1
"""
word = input('Enter a word: ')
for line in file:
    if word == line.strip():
        print('It is in the dictionary')
        """

#2
"""
L = []
for line in file:
    L.append(line.strip())
num = int(input('Enter a number: '))
if num >= 0 and num <= len(L):
    print(L[num])
else:
    print('not in range')
    """
"""
#3
file2 = open('HW6.py')
for line2 in file2:
    line2 = line2.strip()
    line2 += '!'
    print(line2)
"""

#4
number = 0
letter = input('Enter a letter: ')
FinalWord = ''
for line in file:
    line = line.strip()
    if line.count(letter) > number:
        number = line.count(letter)
        FinalWord = line
print(FinalWord)

