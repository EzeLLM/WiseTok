n = int(input())
from collections import deque
queue = deque()

for i in range(n) :
    queue.append(i+1)

while len(queue) > 1 :
    queue.popleft()
    n = queue[0]
    queue.popleft()
    queue.append(n)

print(queue[0])


# 숏 코딩
#
# n = int(input())
# from collections import deque
# queue = deque(i for i in range(1, n+1))
#
# while True :
#     if len(queue) == 1:
#         break
#
#     queue.popleft()
#     queue.append(queue.popleft())
#
# print(queue.pop())
