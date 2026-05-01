"""
240. Search a 2D Matrix II


Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
Example:

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.

"""

# Result AC 64ms 11.87%
# Could reduce compare steps to reduce time
# 关键是是否能知道每次比较时，是从下一行上来的还是从左边走过去的

class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        if not matrix:
        	return False
        if not matrix[0]:
        	return False

        max_row = 0
        for row in matrix:
        	# print(matrix.index(row),row,row[0])
        	if row[0] > target and row is matrix[0]:
        		# print('False')
        		return False 
        	elif row[0] == target:
        		# print('Bang!')
        		return True
        	elif row[0] < target and matrix.index(row) == len(matrix) - 1:
        		# print('Matrix Max')
        		max_row = len(matrix) - 1
        	elif row[0] < target:
        		# print('Going down')
        		pass
        	elif row[0] > target:
        		# print('Max reached')
        		max_row = matrix.index(row) - 1
        		break

        i = max_row
        j = 0
        k = len(matrix[0]) - 1
        # print('max_row: ',i)
        while True:
        	if matrix[i][j]:
        		# print('i',i,'j',j, matrix[i][j])
        		if matrix[i][j] == target:
        			# print('Found!')
        			return True
        		elif matrix[i][j] < target and j == k:
        			return False
        		elif matrix[i][j] < target:
        			# print('Goding right')
        			j += 1
        		elif matrix[i][j] > target:
        			# print('Going up')
        			i -= 1
        			j = 0
        			if i < 0:
        				return False
        	else:
        		return False


if __name__ == '__main__':
    from time import time
    sol = Solution()
    t = time()
    # ans = sol.searchMatrix([[1,   4,  7, 11, 15], [2,   5,  8, 12, 19], [3,   6,  9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]], 20)     
    ans = sol.searchMatrix([[]],1)
    print('ans: %s\ntime: %.3fms' % (ans, ((time() - t)) * 1000))             			
