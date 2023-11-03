def knn(data, profile, k):
    print(k)
    distances = {}
    length = profile.shape[1]
    for x in range(len(data)):
        dist = euclideanDistance(profile, data.iloc[x], length)   
        distances[x] = dist[0]
    sort_dist = sorted(distances.items(), key=lambda x: x[1]) 
    neigh = []
    for x in range(k):
        neigh.append(sort_dist[x][0])
    freq = {}
    for x in range(len(neigh)):
        res = data.iloc[neigh[x]][-1]
        if res in freq:
            freq[res] += 1
        else:
            freq[res] = 1
    sort_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return(sort_freq, neigh)