#import numpy as np

print(
    '''
    built-in open function creates a Python file object.
    this Python file object serves as a link to the file on the machine.
    '''
)

# r: read only mode
# input = open('iris.data', 'r')
# w: write only mode. create file if not exists
# output_float = open('iris.csv', 'w')
# for line in input:
#     line = line.replace('Iris-setosa', '1')
#     line = line.replace('Iris-versicolor', '2')
#     line = line.replace('Iris-virginica', '3')
#     output_float.write(line)
# input.close()
# output_float.close()

# input_float = open('iris.csv', 'r')
# iris_data = np.loadtxt(input_float, delimiter=',')
# input_float.close()
# print(iris_data)

# open file in read only mode: r is default.
READMEmd = open('README.md')
READMEmd_contentasstring = READMEmd.read()

tmp_file = open('tmp.file', 'w')
tmp_file.write('io.py: code')
# flush output buffer without closing
tmp_file.flush()
tmp_file.close()
