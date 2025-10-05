def canSum(sum,a,memo):
    if sum in memo.keys():
        return memo[sum]
    if (sum<0):
        return False
    if (sum==0):
        return True
    for x in a:
        if(canSum(sum-x,a,memo)==True):
            memo[sum]=True
            return True
    memo[sum]=False
    return False
memo={}   
print(canSum(7,[2,3,4,7],memo))