def No2Sort(arr):
    count = 0
    def ms(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = ms(arr[:mid])
        right = ms(arr[mid:])
        return merge(left, right)
    def merge(left, right):
        nonlocal count
        result = []
        l, r = 0, 0
        while l < len(left) or r < len(right):
            if r >= len(right) or (l < len(left) and left[l][1] <= right[r][1]):
                result.append(left[l])
                count += r
                l += 1
            else:
                result.append(right[r])
                r += 1
        return result
    ms(list(enumerate(arr)))
    return count

if __name__ == '__main__':
    n=int(input())
    arr = [int(x) for x in input().split()]
    arr=arr[::-1] 
    print(No2Sort(arr))