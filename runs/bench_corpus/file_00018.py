#!/usr/bin/env python3

import itertools

nums = []
with open("input.txt") as f:
    for line in f:
        nums.append(int(line.strip()))

ans = 0
for sz in range(len(nums)):
    for comb in itertools.combinations(nums, sz):
        if sum(comb) == 150:
            ans += 1
print(ans)

