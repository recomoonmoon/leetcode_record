from typing import  *
from collections import defaultdict, deque
from bisect import bisect_right,bisect_left
import math
import heapq
from heapq import heappop, heapify, heappush, heappushpop, heapreplace



class Solution:

    #3516
    def findClosest(self, x: int, y: int, z: int) -> int:
        if abs(x-z) < abs(y-z):
            return 1
        elif abs(x-z) > abs(y-z):
            return 2
        else:
            return 0
    #2749
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        def compute(num):
            temp = 1
            while temp * 2 <= num:
                temp *= 2

            accu = 0
            while num > 0:
                num -= temp
                accu += 1
                while temp > num:
                    temp //= 2
            return accu

        time = 1
        while True and time < 1000:
            if time >= compute(num1 - time * num2) and compute(num1 - time * num2) > 0 and time <= num1 - time * num2:
                return time
            time += 1
        return -1
    #845
    def longestMountain(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        枚举中间
        """
        mid = 0
        op = 0
        ed = 0
        n = len(arr)
        ans = 0
        while mid < n:
            while op-1 >= 0 and arr[op-1] < arr[op]:
                op -= 1
            while ed+1 < n and arr[ed+1] < arr[ed]:
                ed += 1
            ans = max(ans, ed-op + 1)
            mid = ed + 1
        return  ans


