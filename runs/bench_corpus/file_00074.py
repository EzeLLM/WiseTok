import read_file


def field_to_str():
    '''(list) -> (str)
    Converts list into str and writes it in a file
    0 is for ' ' 1 is for  '*' 2 s for 'X'
    '''
    board = read_file.read_file('field.txt')
    board_str1 = ""
    board_str = []
    for i in board:
        if 0 in i:
            board_str1 += " "
        elif 1 in i:
            board_str1 += "*"
        elif 2 in i:
            board_str1 += "X"
    for i in range(0, 91, 10):
        board_str.append(board_str1[i:i + 10])
    print('Do you want to see a field here', end='')
    print(' or you want to write it in a file field1.txt?')
    control = int(input('Print 1 to see a field or 2 to write in a file '))
    if control == 1:
        for i in board_str:
            print(i)
    elif control == 2:
        with open('field1.txt', 'w') as file:
            for i in board_str:
                file.write(i + '\n')
    else:
        print('You are cheating!')
field_to_str()