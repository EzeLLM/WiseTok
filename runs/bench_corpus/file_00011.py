def cubic(x):
    return x**3

def square(x):
    return x**2

def eval_f_0123(f):
    ''' Given parameter f, evaluate the function f=f(x) 
        at positions x=0, x=1, x=2 and x=3 and 
        return the list [f(0), f(1), f(2), f(3)] '''
        
    return [f(x) for x in [0, 1, 2, 3]]

print(eval_f_0123(cubic))
print(eval_f_0123(square))