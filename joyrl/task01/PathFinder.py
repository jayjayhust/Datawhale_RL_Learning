class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        计算从左上角到右下角有多少种独特的路径。

        Args:
            m (int): 网格的行数。
            n (int): 网格的列数。

        Returns:
            int: 从左上角到右下角的独特路径数量。

        """
        # 初始化边界条件
        f = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]  # 第一行和第一列都是1
        print(f)
        
        # 状态转移
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = f[i - 1][j] + f[i][j - 1]
        return f[m - 1][n - 1]

if __name__ == '__main__':
    s = Solution()
    res = s.uniquePaths(3, 7)
    print(res)