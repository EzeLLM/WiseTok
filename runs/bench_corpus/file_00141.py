def gcd(n1, n2):
    max_range = n1
    if n2 < n1:
        max_range = n2
    gcd_result = 0

    for i in range(1, max_range + 1):
        if n1 % i == 0 and n2 % i == 0:
            gcd_result = i

    return gcd_result

def main():
    token = input('Enter 2 number: ')
    (n1, n2) = token.split()
    print('gcd(%s, %s) = %d' % (
        n1, 
        n2, 
        gcd(
            int(n1), 
            int(n2)
        )
    ))

if __name__ == "__main__":
    main()