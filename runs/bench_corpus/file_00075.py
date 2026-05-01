# Enter your code here. Read input from STDIN. Print output to STDOUT
#mukul jain
T = raw_input()

for S in range(int(T)):
    count = 0
    N = raw_input()
    for numbers in N:
        if int(numbers)!=0 and int(N) % int(numbers) == 0:
            count = count + 1
      
    print count       
            