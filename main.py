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

    def countUnguarded(self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]) -> int:
        record = [[0 for _ in range(n)]for _ in range(m)]

        def guard_dfs(x, y):
            for temp_y in range(y+1, n, 1):
                #这个位置有守卫or墙就可以略过了
                if record[x][temp_y] == -1:
                    break
                #这个位置在纵向上横着过去已经有守卫纵向盯着
                elif record[x][temp_y] == 2 or record[x][temp_y] == 3:
                    break
                #这个位置有人横向盯着
                elif record[x][temp_y] == 1:
                    record[x][temp_y] = 3
                else:
                    record[x][temp_y] = 2

        for w in walls:
            record[w[0]][w[1]] = -1
        for g in guards:
            x = g[0]
            y = g[1]
            record[x][y] = -1
            guard_dfs(x, y)
        ans = 0
        for i in range(m):
            for j in range(n):
               ans += 1 if record[i][j] == 0 else 0
        return  ans

    def mostPoints(self, questions: List[List[int]]) -> int:
        n = len(questions)
        dp = [[0, 0] for _ in range(n)]
        """
        dp[i][0]代表如果在i处于能解决问题的状态，所可以获得的最大分数
        dp[i][1]代表如果在i处于不能解决问题的状态，所可以获得的最大分数
        """

        for idx, question in enumerate(questions):
            if idx == 0:
                dp[idx][0] = 0
                dp[idx][1] = question[0]
            else:
                dp[idx][0] = max(dp[idx-1][0], dp[idx][0])
                dp[idx][1] = max(dp[idx-1][1], dp[idx][0] + question[0])

            next_res_idx = idx + question[1] + 1
            if next_res_idx < n:
                dp[next_res_idx][0] = max(dp[next_res_idx][0], dp[idx][0] + question[0])


        return max(dp[-1])

    def triangleNumber(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        nums.sort()
        for i in range(0, n-2, 1):
            for j in range(i+1, n-1, 1):
                maxlen = nums[i] + nums[j]
                target = bisect_left(nums, maxlen)
                if target > j and nums[i] > 0:
                    ans += target - j - 1
        return ans

    def splitArray(self, nums: List[int]) -> int:
        split_idx = 0
        n = len(nums)
        for i in range(1, n, 1):
            if nums[i] <= nums[i-1]:
                split_idx = i
                break
        if split_idx == 0:
            split_idx = n
        for i in range(split_idx, n, 1):
            if i > 0 and nums[i] > nums[i-1]:
                return -1

        if split_idx == n:
            return abs(sum(nums[:n-1]) - nums[-1])
        elif split_idx == 1:
            return abs(sum(nums[1:]) - nums[0])
        else:
            ans1 = abs(sum(nums[:split_idx]) - sum(nums[split_idx:]))
            ans2 = abs(sum(nums[:split_idx-1]) - sum(nums[split_idx-1:]))
            return min(ans1, ans2)

    def openLock(self, deadends: List[str], target: str) -> int:
        if target == "0000":
            return 0
        record = defaultdict(int)
        visit = defaultdict(int)
        for d in deadends:
            visit[f"{d[0]}{d[1]}{d[2]}{d[3]}"] = 1
        queue = [[0,0,0,0,0]]
        while queue:
            curr = queue.pop(0)
            a,b,c,d,time = curr[0], curr[1], curr[2], curr[3], curr[4]
            if f"{a}{b}{c}{d}" == target:
                record[f"{a}{b}{c}{d}"] = time
                break
            if visit[f"{a}{b}{c}{d}"]:
                continue
            visit[f"{a}{b}{c}{d}"] = 1
            record[f"{a}{b}{c}{d}"] = time
            queue.append([(a+1)%10,b,c,d,time+1])
            queue.append([(a-1)%10,b,c,d,time+1])
            queue.append([a,(b+1)%10,c,d,time+1])
            queue.append([a,(b-1)%10,c,d,time+1])
            queue.append([a,b,(c+1)%10,d,time+1])
            queue.append([a,b,(c-1)%10,d,time+1])
            queue.append([a,b,c,(d+1)%10,time+1])
            queue.append([a,b,c,(d-1)%10,time+1])
        if record[target]:
            return record[target]
        else:
            return -1

    def reachNumber(self, target: int) -> int:
        target = abs(target)
        accu = 0
        t = 1
        while True:
            accu += t
            t += 1
            if accu >= target:
                if accu - target % 2 == 0:
                    return t-1

    def triangularSum(self, nums: List[int]) -> int:
        # 1
        # 1 1
        # 1 2 1
        # 1 3 3 1
        # 1 4 6 4 1
        # 1 5 10 10 5 1
        # 1 6 15 20 15 6 1
        def bfs(nums,  target):
            if target == len(nums):
                return nums
            lister = []
            for i in range(len(nums)):
                if i == 1 or i == len(nums) - 1:
                    lister.append(1)
                else:
                    lister.append(nums[i-1] + nums[i])
            return bfs(lister, target)

        weight = bfs([1], len(nums))
        accu = 0
        for i in range(len(nums)):
            accu = (accu + weight[i] * nums[i]) % 10
        return accu

    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        # 初始化全 1 的矩阵
        grid = [[1] * n for _ in range(n)]
        for x, y in mines:
            grid[x][y] = 0

        left = [[0] * n for _ in range(n)]
        right = [[0] * n for _ in range(n)]
        up = [[0] * n for _ in range(n)]
        down = [[0] * n for _ in range(n)]

        # 计算 left 和 up
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    left[i][j] = 1 + (left[i][j - 1] if j > 0 else 0)
                    up[i][j] = 1 + (up[i - 1][j] if i > 0 else 0)

        # 计算 right 和 down
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if grid[i][j] == 1:
                    right[i][j] = 1 + (right[i][j + 1] if j < n - 1 else 0)
                    down[i][j] = 1 + (down[i + 1][j] if i < n - 1 else 0)

        ans = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    size = min(left[i][j], right[i][j], up[i][j], down[i][j])
                    ans = max(ans, size)
        return ans

    def slidingPuzzle(self, board: List[List[int]]) -> int:
        def get_val(board):
            board = board[0] + board[1]
            ans = 0
            for i in range(6):
                ans += 10**i * board[i]
            return ans

        mp = defaultdict(int)
        q = [[board, 0]]
        while q:
            temp_board, time = q.pop(0)
            val = get_val(temp_board)
            if val == 54321:
                return time
            elif mp[val]:
                continue
            else:
                mp[val] = time + 1

            idx = -1
            idy = -1
            for i in range(2):
                for j in range(3):
                    if temp_board[i][j] == 0:
                        idx = i
                        idy = j
                        break

            for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                tempx = idx + delta[0]
                tempy = idy + delta[1]
                if -1< tempx < 2 and -1< tempy< 3:
                    tb = [temp_board[0].copy(), temp_board[1].copy()]
                    tb[idx][idy] = tb[tempx][tempy]
                    tb[tempx][tempy] = 0
                    q.append([tb, time+1])
        return -1

    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        ans = 0
        water = numBottles
        while water:
            ans += water
            water = numBottles // numExchange
            numBottles = numBottles // numExchange + numBottles % numExchange
        return ans

    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
        while tx > 0 and ty > 0:
            if tx == sx and ty == sy:
                return True
            if tx == ty:
                return sx == 0 and sy == 0
            else:
                if tx > ty:
                    tx, ty = ty, tx-ty
                else:
                    tx, ty = ty - tx, tx
        return tx == sx and ty == sy

    def maxBottlesDrunk(self, numBottles: int, numExchange: int) -> int:
        drunked = 0
        empty = 0
        flag = True
        while numBottles:
            drunked += numBottles
            numBottles = 0
            empty += numBottles
            if empty >= numExchange:
                numBottles += 1
                empty -= numExchange
                numExchange += 1
        return drunked

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m = len(heights)
        n = len(heights[0])
        record = [[0] * n for _ in range(m)]

        def dfs(x, y, mode=1):
            if -1 < x < m and -1 < y < n:
                if record[x][y] == mode or record[x][y] == 3:
                    return
                else:
                    record[x][y] += mode
                    for delta in [[-1,0], [1,0], [0,1], [0,-1]]:
                        tempx = x+delta[0]
                        tempy = y+delta[1]
                        if -1 < tempx < m and -1 < tempy < n:
                            if heights[tempx][tempy] >= heights[x][y]:
                                dfs(tempx, tempy, mode)
        for i in range(n):
            dfs(0, i, 1)
            dfs(m-1, i, 2)
        for i in range(m):
            dfs(i, 0, 1)
            dfs(i, n-1, 2)

        accu = []
        for i in range(m):
            for j in range(n):
                if record[i][j] == 3:
                    accu.append([i,j])
        return accu

    def minTime(self, skill: List[int], mana: List[int]) -> int:
        """
        1 2 2 3 1 1
        2 4 4 6 2 2
        1 2 2 3 1 1
        """
        n = len(skill)
        sum_mana = sum(mana)
        pre_mana = sum_mana - mana[-1]
        def compute(pre_sk, post_sk):
            if pre_sk <= post_sk:
                return 0
            else:
                pre_cost = pre_sk * sum_mana
                post_cost = post_sk * pre_mana
                if post_cost >= pre_cost:
                    return 0
                else:
                    return pre_cost - post_cost


        pre = -1
        accu = 0
        for sk in skill:
            if pre < 0:
                pre = sk
                continue
            else:
                bubble = compute(pre, sk)
                accu += max(pre * mana[0], bubble)
                pre = sk

        accu += sum_mana * skill[-1]
        return accu

    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        mp = defaultdict(int)
        for p in power:
            mp[p] += 1
        keys = list(mp.keys())
        n = len(power)
        dp=[[0,0]for _ in range(len(keys))]
        dp[0][1] = keys[0] * mp[keys[0]]
        for i in range(1, len(keys)):
            now_key = keys[i]
            pre_key = keys[i-1]
            if now_key - pre_key > 2:
                dp[i][0] = max(dp[i-1][0], dp[i-1][1])
                dp[i][1] = max(dp[i-1]) + mp[now_key] * now_key
            else:
                free_key_idx = i-2
                while free_key_idx >= 0 and now_key - keys[free_key_idx] <= 2:
                    free_key_idx -= 1
                if free_key_idx > -1:
                    dp[i][0] = max(dp[i-1][0], dp[free_key_idx][1])
                    dp[i][1] = max(dp[free_key_idx]) + mp[now_key] * now_key
                else:
                    dp[i][0] = dp[i - 1][0]
                    dp[i][1] = mp[now_key] * now_key
        return max(i[1] for i in dp)

    def expressiveWords(self, s: str, words: List[str]) -> int:
        ans = 0
        def extract_word(word:str):
            stack = []
            for c in word:
                if not stack:
                    stack.append([c, 1])
                else:
                    if c == stack[-1][0]:
                        stack[-1][1] += 1
                    else:
                        stack.append([c, 1])
            return stack

        template = extract_word(s)
        n = len(template)
        for word in words:
            word_stack = extract_word(word)
            if len(word_stack) != len(template):
                continue
            else:
                flag = True
                for i in range(n):
                    if word_stack[i][0] != template[i][0] or (template[i][1] > word_stack[i][1] and template[i][1] < 3) or template[i][1] < word_stack[i][1]:
                        flag = False
                        break
                if flag:
                    ans += 1
        return ans
    def lengthLongestPath(self, input: str) -> int:
        stack = []
        lines = input.split(r"\n")
        max_length = 0
        def count_t(s:str):
            n = len(s)
            ans = 0
            for i in range(0, n-1, 2):
                if s[i:i+2] == r'\t':
                    ans += 1
                else:
                    return ans
            return ans
        for line in lines:
            if not stack:
                stack.append([0, len(line)])
            else:
                t_num = count_t(line)
                if t_num > stack[-1][0]:
                    stack.append([t_num, len(line)-2*t_num])
                elif t_num == stack[-1][0]:
                    stack[-1] = [t_num, len(line)-2*t_num]
                else:
                    while stack[-1][0] >= t_num:
                        stack.pop()
                    stack.append([t_num, len(line)-2*t_num])

                if "." in line:
                    temp = 0
                    for i in stack:
                        temp += i[1] + 1
                    temp -= 1
                    max_length = max(max_length, temp)
            print(stack)
            print(f"Max Length: {max_length}")
        return max_length

    def removeAnagrams(self, words: List[str]) -> List[str]:
        stack = []
        def get_dict(word:str):
            mp = defaultdict(int)
            for c in word:
                mp[c] += 1
            return mp
        def juege(mp1, mp2):
            if len(mp1) != len(mp2):
                return False
            for k in mp1.keys():
                if k not in mp2 or mp1[k] != mp2[k]:
                    return False
            return True
        ans = []
        for word in words:
            if not stack:
                stack.append(get_dict(word))
                ans.append(word)
            else:
                temp = get_dict(word)
                if juege(temp, stack[-1]):
                    continue
                else:
                    stack.append(temp)
                    ans.append(word)
        return ans

    def findSmallestInteger(self, nums: List[int], value: int) -> int:
        record = [0 for _ in range(value)]
        for num in nums:
            record[num % value] += 1
        k = min(record)
        base = k * value - 1
        for i in range(value):
            if record[i % value] > k:
                base += 1
            else:
                break
        return base + 1

    def minCost(self, arr: List[int], brr: List[int], k: int) -> int:
        n = len(arr)
        ans = sum([abs(arr[i] - brr[i]) for i in range(n)])
        arr.sort()
        brr.sort()
        ans = min(ans, k + sum([abs(arr[i] - brr[i]) for i in range(n)]))
        return ans

    def maximumAmount(self, coins: List[List[int]]) -> int:
        m = len(coins)
        n = len(coins[0])
        dp = [[[0,0,0]for _ in range(n)] for _ in range(m)]


        for i in range(m):
            for j in range(n):
                if i == j == 0:
                    dp[0][0][0] = coins[0][0]
                    dp[0][0][1] = abs(coins[0][0])
                    dp[0][0][2] = abs(coins[0][0])
                else:
                    if i == 0:
                        dp[i][j][0] = dp[i][j-1][0] + coins[i][j]
                        dp[i][j][1] = max(dp[i][j-1][0] + abs(coins[i][j]), dp[i][j-1][1] + coins[i][j])
                        dp[i][j][2] = max(dp[i][j-1][1] + abs(coins[i][j]), dp[i][j-1][2] + coins[i][j])
                    elif j == 0:
                        dp[i][j][0] = dp[i-1][j][0] + coins[i][j]
                        dp[i][j][1] = max(dp[i-1][j][0] + abs(coins[i][j]), dp[i-1][j][1] + coins[i][j])
                        dp[i][j][2] = max(dp[i-1][j][1] + abs(coins[i][j]), dp[i-1][j][2] + coins[i][j])
                    else:
                        dp[i][j][0] = max(dp[i-1][j][0], dp[i][j-1][0]) + coins[i][j]
                        dp[i][j][1] = max([max(dp[i-1][j][0], dp[i][j-1][0]) + abs(coins[i][j]), dp[i-1][j][1] + coins[i][j], dp[i][j-1][1]+coins[i][j]])
                        dp[i][j][2] = max([max(dp[i-1][j][1], dp[i][j-1][1]) + abs(coins[i][j]), dp[i-1][j][2] + coins[i][j], dp[i][j-1][2]+coins[i][j]])

        return dp[-1][-1][2]

    def furthestDistanceFromOrigin(self, moves: str) -> int:
        record = 0
        accu = 0
        ans = 0
        for m in moves:
            if m == "L":
                record -= 1
            elif m == "R":
                record += 1
            else:
                accu += 1
            ans = max(accu + abs(record), ans)
        return ans
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def dfs(i: int, mask: int, changed: bool) -> int:
            if i == len(s):
                return 1

            # 不改 s[i]
            bit = 1 << (ord(s[i]) - ord('a'))
            new_mask = mask | bit
            if new_mask.bit_count() > k:
                # 分割出一个子串，这个子串的最后一个字母在 i-1
                # s[i] 作为下一段的第一个字母，也就是 bit 作为下一段的 mask 的初始值
                res = dfs(i + 1, bit, changed) + 1
            else:  # 不分割
                res = dfs(i + 1, new_mask, changed)
            if changed:
                return res

            # 枚举把 s[i] 改成 a,b,c,...,z
            for j in range(26):
                new_mask = mask | (1 << j)
                if new_mask.bit_count() > k:
                    # 分割出一个子串，这个子串的最后一个字母在 i-1
                    # j 作为下一段的第一个字母，也就是 1<<j 作为下一段的 mask 的初始值
                    res = max(res, dfs(i + 1, 1 << j, True) + 1)
                else:  # 不分割
                    res = max(res, dfs(i + 1, new_mask, True))
            return res

        return dfs(0, 0, False)

    def finalValueAfterOperations(self, operations: List[str]) -> int:
        num = 0
        for operation in operations:
            if operation[0] == "-" or operation[2] == "-":
                num -= 1
            else:
                num += 1
        return  num

    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()


    def findMaximumScore(self, nums: List[int]) -> int:
        stack = []
        ans = 0
        for idx, num in enumerate(nums):
            if not stack:
                stack.append([num, idx])
            else:
                if num > stack[0][0]:
                    ans += (idx - stack[-1][1]) * stack[-1][0]
                    stack.pop()
                    stack.append([num, idx])
                else:
                    pass
        if stack:
            ans += stack[-1][0] * (len(nums) - 1 - stack[-1][1])
        return ans

    def removeSubstring(self, s: str, k: int) -> str:
        stack = []
        for c in s:
            if not stack:
                stack.append([c, 1])
            else:
                if c == stack[-1][0]:
                    stack.append([c, stack[-1][1] + 1])
                else:
                    stack.append([c, 1])
                if c == ")" and stack[-1][1] == k and len(stack) >= 2*k and stack[-k-1][0] == "(" and stack[-k-1][1] >= k:
                    for _ in range(2*k):
                        stack.pop()
        return "".join([i[0] for i in stack])

    def distinctPoints(self, s: str, k: int) -> int:
        record = [0, 0]
        mp = defaultdict(str)
        def compute(c):
            if c == "U":
                return [0, 1]
            elif c == "D":
                return [0, -1]
            elif c == "L":
                return [1, 0]
            else:
                return [-1, 0]

        for i in range(k):
            temp = compute(s[i])
            record[0] += temp[0]
            record[1] += temp[1]
        mp[f"{record[0]}-{record[1]}"] = 1
        for i in range(k, len(s), 1):
            temp = compute(s[i-k])
            record[0] -= temp[0]
            record[1] -= temp[1]
            temp = compute(s[i])
            record[0] += temp[0]
            record[1] += temp[1]
            mp[f"{record[0]}-{record[1]}"] = 1

        return len(mp)

    def climbStairs(self, n: int, costs: List[int]) -> int:
        dp = [0 for _ in range(n)]
        dp[0] = costs[0] + 1
        if n > 1:
            dp[1] = min(dp[0] + 1,  4) + costs[1]
        if n > 2:
            dp[2] = min(9, dp[0]+4, dp[1]+1) + costs[2]

        for i in range(3, n, 1):
            dp[i] = min([dp[i-1] + 1,  dp[i-2] + 4, dp[i-3] + 9]) + costs[i]

        return dp[-1]

s = Solution()
print(s.removeSubstring(s = "((()))()()()", k = 1))