from typing import  *
from collections import defaultdict, deque
from bisect import bisect_right,bisect_left
import math
import heapq
from heapq import heappop, heapify, heappush, heappushpop, heapreplace
from collections import Counter

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

    def minOperations(self, queries: List[List[int]]) -> int:
        def compute(left, right):
            k = 0
            ans = 0
            temp = 0
            while 4**k - 1 < left:
                k+=1
            while 4**k - 1 < right or left <= right:
                num = min(4**k-1, right) - left + 1
                ans += k * num // 2
                if k * num % 2 == 1:
                    temp += 1
                left = 4**k
                k+=1
            ans += temp // 2 + temp % 2
            return ans
        ans = 0
        for query in queries:
            op = query[0]
            ed = query[1]
            ans += compute(op, ed)
        return ans

    def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:

        def compute(s):
            temp = ""
            base = ord('a')
            for c in s:
                temp += str(ord(c) - base)
            return int(temp)
        print(compute(firstWord), compute(secondWord), compute(targetWord))
        return compute(firstWord) + compute(secondWord) == compute(targetWord)

    def addDigits(self, num: int) -> int:
        while num > 9:
            temp = 0
            while num:
                temp += num % 10
                num //= 10
            num = temp
        return num

    def arrayRankTransform(self, arr: List[int]) -> List[int]:
        l = list(set(arr))
        l.sort()
        mp = {}
        for idx in range(len(l)):
            mp[l[idx]] = idx + 1
        arr = [mp[i] for i in arr]
        return arr

    def sumZero(self, n: int) -> List[int]:
        if n % 2 == 0:
            k = n // 2
            arr = list(range(1, k + 1, 1))
            arr = arr + [-i for i in arr]
        else:
            k = (n - 1) // 2
            arr = list(range(1, k + 1, 1))
            arr = arr + [-i for i in arr]
            arr.append(0)
        return arr

    def perfectPairs(self, nums: List[int]) -> int:
        """
        a = nums[i]，b = nums[j]。那么：
        min(|a - b|, |a + b|) <= min(|a|, |b|)
        意味着 如果 a和b同号，|a - b| <= min(|a|, |b|) 大 - 小 <= 小 -》 大 <= 2*小
              如果 a和b异号 |a + b| <= min(|a|, |b|) 大 - 小 <= 小 -》 大 <= 2*小
        max(|a - b|, |a + b|) >= max(|a|, |b|)
        意味着 如果 a和b同号，|a + b| >= max(|a|, |b|) 大 + 小 >= 大 -》 符合条件
              如果 a和b异号 |a - b| >= max(|a|, |b|) 大 + 小 >= 小 -》 符合条件
        """
        n = len(nums)
        ans = 0
        for i in range(n-1):
            for j in range(i+1, n):
                a = max(abs(nums[i]), abs(nums[j]))
                b = min(abs(nums[i]), abs(nums[j]))
                if a <= 2*b:
                    ans+=1
        return ans

    def minimumLength(self, s: str) -> int:
        mp = defaultdict(list)
        ans = 0
        for idx, c in enumerate(s):
            mp[c].append(idx)
        for key, val in mp.items():
            if len(val) >= 3:
                ans += 1 if len(val) % 4 == 1 else 2
            else:
                ans += len(val)
        return ans

    def queryResults(self, limit: int, queries: List[List[int]]) -> List[int]:
        color_idx = defaultdict(dict)
        idx_color = defaultdict(int)
        ans = []
        color_num = 0
        for idx, color in queries:
            if idx not in idx_color:
                idx_color[idx] = color
                if len(color_idx[color]) == 0:
                    color_num += 1
                color_idx[color][idx] = 1
                ans.append(color_num)
            else:
                old_color =  idx_color[idx]
                if len(color_idx[old_color]) == 1:
                    del color_idx[old_color]
                    color_num -= 1
                else:
                    del color_idx[old_color][idx]
                idx_color[idx] = color
                if len(color_idx[color]) == 0:
                    color_num += 1
                color_idx[color][idx] = 1
                ans.append(color_num)
        return ans

    def minimumTeachings(self, n: int, languages: List[List[int]], friendships: List[List[int]]) -> int:
        record = [0 for _ in range(n+1)]

        for idx, u, v in enumerate(friendships):
            lenu = len(languages[u])
            lenv = len(languages[v])
            unit_languages = set(languages[u] + languages[v])
            if len(unit_languages) < lenv + lenu:
                continue
            else:
                for i in range(n+1):
                    if i not in unit_languages:
                        record[i] += 2
                    else:
                        record[i] += 1
        ans = 0
        minnum = 9999999999999999
        for lan, num in enumerate(record):
            if num < minnum:
                ans = lan
                minnum = num
        return ans

    def sortVowels(self, s: str) -> str:
        meta = []
        mp = defaultdict(int)
        ans = ["" for _ in range(len(s))]
        for k in "aeiouAEIOU":
            mp[k] = 1
        for idx,c in enumerate(s):
            if mp[c]:
                meta.append(c)
            else:
                ans[idx] = c
        meta.sort(key=lambda x:ord(x))
        for idx, c in enumerate(s):
            if mp[c] == 0:
                ans[idx] = meta.pop(0)
        return "".join(ans)

    def largestPalindromic(self, num: str) -> str:
        record = [0 for _ in range(10)]
        for n in num:
            record[int(n)] += 1
        ans = ""
        maxnum = ""
        for i in range(9, -1, -1):
            if maxnum == "" and record[i] % 2 == 1:
                maxnum = str(i)
            ans += str(i) * (record[i]//2)
        ans += "*"
        ans = ans.strip("0")
        ans = ans.strip("*")
        return ans + maxnum + ans[::-1]

    def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        n = len(passingFees)
        self.min_money_cost = 999999999999999999999
        self.min_time_cost = 999999999999999999999
        self.visited = [0 for _ in range(n)]
        self.mp = defaultdict(dict)
        for e in edges:
            self.mp[e[0]][e[1]] = e[2]
            self.mp[e[1]][e[0]] = e[2]
        def dfs(root, time_cost, money_cost):
            self.visited[root] = 1
            if root == n-1:
                self.min_money_cost = money_cost
                return
            for neighbor in self.mp[root]:
                if self.visited[neighbor] == 0 and time_cost + self.mp[root][neighbor] <= maxTime and money_cost + passingFees[neighbor] < self.min_money_cost:
                    dfs(neighbor, time_cost + self.mp[root][neighbor], money_cost + passingFees[neighbor])
                    self.visited[neighbor] = 0

        dfs(0, 0, passingFees[0])
        if self.min_money_cost == 999999999999999999999:
            return -1
        else:
            return self.min_money_cost

    def maxFreqSum(self, s: str) -> int:
        a1 = 0
        a2 = 0
        mp = defaultdict(int)
        for char in s:
            if char in "aoeiu":
                mp[char] += 1
                a1 = max(a1, mp[char])
            else:
                mp[char] += 1
                a2 = max(a2, mp[char])
        return a1 + a2

    def spellchecker(self, wordlist: List[str], queries: List[str]) -> List[str]:
        word_temps = defaultdict(str)
        for idx, word in enumerate(wordlist):
            temp = ""
            word1 = word.lower()
            for c in word1:
                if c in "aeiou":
                    temp += "1"
                else:
                    temp += c
            word_temps[temp] = word
        ans = []
        for idx, word in enumerate(queries):
            temp = ""
            word = word.lower()
            for c in word:
                if c in "aeiou":
                    temp += "1"
                else:
                    temp += c
            ans.append(word_temps[temp])
        return ans

    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        words = text.split()
        mp = defaultdict(int)
        for char in brokenLetters:
            mp[char] = 1
        ans = 0
        for word in words:
            flag = 1
            for char in word:
                if mp[char] == 1:
                    flag = 0
                    break
            ans += flag
        return ans

    def digitSum(self, s: str, k: int) -> str:
        while len(s) > k:
            op = 0
            temp = ""
            while op < len(s):
                chunk = s[op:min(len(s), op+k)]
                op += k
                chunk = str(sum([int(i) for i in chunk]))
                temp += chunk
            s = temp
        return s

    def bestHand(self, ranks: List[int], suits: List[str]) -> str:
        suit_mp = defaultdict(int)
        rank_mp = defaultdict(int)
        for r in ranks:
            rank_mp[r] += 1
        for s in suits:
            suit_mp[s] += 1

        for k,v in suit_mp.items():
            if v == 5:
                return "Flush"
        if max(list(rank_mp.values())) >= 3:
            return "Three of a Kind"
        elif max(list(rank_mp.values())) >= 2:
            return "Pair"
        return "High Card"

    def evaluate(self, s: str) -> int:
        # Step 1: tokenize
        stack = []
        for c in s.replace(" ", ""):  # 去掉空格
            if c.isdigit():
                if stack and stack[-1].isdigit():
                    stack[-1] += c
                else:
                    stack.append(c)
            else:
                stack.append(c)

        # Step 2: handle * /
        new_stack = []
        i = 0
        while i < len(stack):
            token = stack[i]
            if token == "*":
                prev = int(new_stack.pop())
                i += 1
                next_num = int(stack[i])
                new_stack.append(prev * next_num)
            elif token == "/":
                prev = int(new_stack.pop())
                i += 1
                next_num = int(stack[i])
                new_stack.append(int(prev / next_num))  # 向 0 取整
            else:
                new_stack.append(token)
            i += 1

        # Step 3: handle + -
        res = int(new_stack[0])
        i = 1
        while i < len(new_stack):
            op = new_stack[i]
            num = int(new_stack[i + 1])
            if op == "+":
                res += num
            else:
                res -= num
            i += 2

        return res

    def getMinDistSum(self, positions: List[List[int]]) -> float:
        xs = [i[0] for i in positions]
        ys = [i[1] for i in positions]
        minx = min(xs)
        maxx = max(xs)
        miny = min(ys)
        maxy = max(ys)
        ans = 999999999999999
        record = []
        for i in range(minx, maxx+1):
            for j in range(miny, maxy+1):
                temp_xs = [(i - ti)**2 for ti in xs]
                temp_ys = [(j - tj)**2 for tj in ys]
                t = math.sqrt(sum([temp_xs[i] + temp_ys[i] for i in range(len(positions))]))
                if t < ans:
                    ans = t
                    record = [i, j]
        print(record)
        return ans

    def minOperations(self, nums: List[int]) -> int:
        def dfs(s:str):
            if len(s) == 0:
                return 0
            elif len(s) == 1:
                return 1
            s_list = s.split("0")
            ans = 0
            for s in s_list:
                if len(s) == 0:
                    continue
                c = Counter(s)
                char = c.most_common(1)[0][0]
                ans += 1
                lister =  [ i for i in  s.split(char) if i]
                for new_s in lister:
                    ans += dfs(new_s)
            return ans
        return dfs("".join([str(i) for i in nums]))

    def maxDistance(self, s: str, k: int) -> int:
        x = [0,0]
        y = [0,0]
        ans = 0
        for idx, c in enumerate(s):
            if c == "N":
                y[0] += 1
            elif c == 'S':
                y[1] += 1
            elif c == "W":
                x[1] += 1
            else:
                x[0] += 1
            minnum = min(x) + min(y)
            if k >= minnum:
                ans = max(ans, max(x) + max(y) + minnum)
            else:
                ans = max(ans, max(x) + max(y) +2*k - minnum)
        return ans

    def uniquePaths(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        if grid[m-1][n-1] == 1:
            return 0
        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0][0] = 1

        for i in range(m):
            for j in range(n):
                if i + j == 0:
                    continue
                else:
                    if i == 0:
                        if grid[i][j] == 0:
                            dp[i][j] += dp[i][j-1]
                        else:
                            if i + 1 < m:
                                dp[i+1][j] += dp[i][j-1]
                    elif j == 0:
                        if grid[i][j] == 0:
                            dp[i][j] += dp[i-1][j]
                        else:
                            if j + 1< n:
                                dp[i][j+1] += dp[i-1][j]
                    else:
                        if grid[i][j] == 0:
                            dp[i][j] += dp[i-1][j] + dp[i][j-1]
                        else:
                            if i+1 < m:
                                #左边来的会变到下面
                                dp[i+1][j] += dp[i][j-1]
                            if j+1 < n:
                                dp[i][j+1] += dp[i-1][j]
        return dp[m-1][n-1]

    def bowlSubarrays(self, nums: List[int]) -> int:
        stack = []
        ans = 0
        for idx, num in enumerate(nums):
            if not stack:
                stack.append([num, idx])
            else:
                while stack and num >= stack[-1][0]:
                    stack.pop()

                if stack:
                    gap = idx - stack[-1][1]
                    if gap > 2:
                        ans += idx - stack[-1][1] + 1
                stack.append([num, idx])
        return ans

    def maxWeight(self, pizzas: List[int]) -> int:
        times = len(pizzas) // 4
        pizzas.sort()
        if times % 2 == 0:
            right = times//2
        else:
            right = times//2 + 1
        ans = 0
        for i in range(right):
            ans += pizzas.pop()
        for i in range(times//2):
            pizzas.pop()
            ans+=pizzas.pop()
        return ans

    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        record = []
        n = len(s)
        for i in range(n):
            if s[i] == "0" and (i == 0 or s[i-1] == "1"):
                record.append(i)
            if s[i] == "0" and (i==n-1 or s[i+1] == "1"):
                record.append(i)



s = Solution()
print(s.minCost(maxTime = 29, edges = [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]], passingFees = [5,1,2,20,20,3]))


class Spreadsheet:

    def __init__(self, rows: int):
        self.rows = rows
        self.sheet = [[0] * 26 for i in range(rows)]

    def setCell(self, cell: str, value: int) -> None:
        self.self[ord(cell[0]) - ord('A')][int(cell[1:]) - 1] = value

    def resetCell(self, cell: str) -> None:
        self.self[ord(cell[0]) - ord('A')][int(cell[1:]) - 1] = 0

    def getValue(self, formula: str) -> int:
        formula = formula[1:]
        formula = formula.split("+")
        record = []
        for f in formula:
            if f.isdigit():
                record.append(int(f))
            else:
                record.append(self.getValue(f))
        return sum(record)

    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        record = []
        n = len(endTime)
        for i in range(n):
            accu = 0 if not record else record[-1]
            record.append(endTime[i] - startTime[i] + accu)
        record.insert(0, 0)
        record.append(record[-1])
        startTime.insert(0, 0)
        startTime.append(eventTime)
        endTime.insert(0, 0)
        endTime.append(eventTime)
        ans = 0
        n = len(record)
        for op in range(n):
            ed = min(n-1, op + k + 1)
            all_length = endTime[ed] - startTime[op]
            length = record[ed] - record[max(0, op-1)]
            ans = max(ans, all_length - length)
        return ans

    def closestTarget(self, words: List[str], target: str, startIndex: int) -> int:
        n = len(words)
        minnum = float("inf")

        for i in range(n):
            if words[i] == target:
                # 环形距离：取正向和反向的最小值
                dist = abs(i - startIndex)
                dist = min(dist, n - dist)
                minnum = min(minnum, dist)

        return -1 if minnum == float("inf") else minnum
