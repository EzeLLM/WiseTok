def StackMenu():
    global choice
    print("Main Menu")
    print("1. Push Data")
    print("2. Pop Data")
    print("3. View stack")
    print("4. Empty stack")
    print("5. Quit")
    choice = int(input("Enter your choice: "))

def ViewStack():
    global max
    i = max - 1
    while(i>=0):
        print ("------")
        if (stack[i]!=0):
            print(stack[i])
        else:
            print("")
        i=i-1

def push(d):
    global top
    if (isFull()):
        print("***ERROR: Stack is FULL")
    else:
        top = top + 1
        stack[top] = d

def pop():
    global bottom, top
    if isEmpty() == True:
        print("***ERROR: Stack is Empty")
    else:
        stack.remove(stack[0])
        stack.insert(7, 0)
        top = top - 1



def isFull():
    global top,max
    if(top == max):
        return True
    else:
        return False

def isEmpty():
    global top
    if (top == -1):
        return True
    else:
        return False

def EmptyStack():
    global stack
    stack = [0,0,0,0,0,0,0,0]

#mainmodule
stack = [0,0,0,0,0,0,0,0]
max = 8
top = -1
bottom = 0
choice = 0
while (choice!=5):
    StackMenu()
    if (choice==1):
        print ("Push Data")
        print ("=========")
        d = int(input("Input your number: "))
        push(d)
    elif (choice==2):
        print ("Pop Data")
        print ("========")
        pop()
        if isEmpty() == True:
            print("...")
        else:
            print ("A number is taken out from the stack")
    elif (choice==3):
        print ("View Stack")
        print ("==========")
        ViewStack()
    elif (choice==4):
        EmptyStack()
        print ("Empty Stack")
        print ("===========")
        print ("Stack has been emptied")

    else:
        print ("Program ended")
        break
