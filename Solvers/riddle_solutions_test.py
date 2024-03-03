def solve_problem_solving_easy(input: tuple) -> list:

    a, k = input
    freq = {}
    for i in a:
        freq[i] = freq.get(i,0) + 1

    ans = [(-freq[key], key) for key in freq]
    ans.sort()
    ans = [pair[1] for pair in ans[:k]]
    return ans

# def solve_problem_solving_medium(input: str) -> str:

#     rep = 0
#     out = ""
#     stk = []
#     for i in input:
#         if i >= '0' and i <= '9':
#             rep*=10
#             rep+=int(i)
#         elif i == '[':
#             stk.append((out, rep))
#             rep = 0
#             out = ""
#         elif i == ']':
#             s, rp = stk.pop()
#             out = s+out*rp
#         else:
#             out+=i
#     return out

#     """
#     This function takes a string as input and returns a string as output.

#     Parameters:
#     input (str): A string representing the input data.

#     Returns:
#     str: A string representing the solution to the problem.
#     """
#     return ''

# def solve_problem_solving_hard(input: tuple) -> int:

#     n, m = input

#     dp = [[0 for _ in range(m)] for _ in range(n)]

#     dp[0][0] = 1
#     for i in range(n):
#         for j in range(m):
#             if i > 0: dp[i][j]+=dp[i-1][j]
#             if j > 0: dp[i][j]+=dp[i][j-1]
#     return dp[n-1][m-1]
#     """
#     This function takes a tuple as input and returns an integer as output.

#     Parameters:
#     input (tuple): A tuple containing two integers representing m and n.

#     Returns:
#     int: An integer representing the solution to the problem.
#     """
#     return 0

print(solve_problem_solving_easy((["pharaoh","sphinx","pharaoh","pharaoh","nile", "sphinx","pyramid","pharaoh",
 "sphinx","sphinx"], 3)))