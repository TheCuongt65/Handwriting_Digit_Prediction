import numpy as np
def myKmeans(X_bar, k):
    J_history = []
    Label_history = []
    C_history = []
    for seed in range(1000):
        c = np.array([0 for i in range(X_bar.shape[1])])
        centroids = np.array([[np.random.randint(1,100) for x in range(X_bar.shape[0])] for i in range(k)])
        cost = 0

        for i in range(1000):
            distances = [np.linalg.norm(X_bar.T - centroid, axis=1) for centroid in centroids]
            
            if k == 1:
                J_history.append(np.mean(distances))
                Label_history.append(c)
                C_history.append(centroids)
                break

            index = np.argmin(distances, axis=0)
            
            if len(set(index)) == k:
                c = index
            else: 
                break
            
            distances_ci_x = np.array([])
            for j in range(k):
                distances_ci_x_j = np.linalg.norm(X_bar.T[np.where(c == j), :][0] - centroids[j], axis=1)
                distances_ci_x = np.concatenate((distances_ci_x, distances_ci_x_j), axis=0)
            # print(distances_ci_x_j.shape)

            for i, centroid in enumerate(centroids):
                centroids[i] = np.mean(X_bar[:,c == i], axis=1)

            if (i != 0) and (abs(cost - np.mean(distances_ci_x)) < 0.005):
                # print(f"Seed at {seed}, Stop at i = {i} and cost = {cost}")
                J_history.append(cost)
                Label_history.append(c)
                C_history.append(centroids)
                break
            else:
                cost = np.mean(distances_ci_x)
                # print(f"Seed at {seed}, Cost at i = {i} is {cost}")
    
    cost = min(J_history)
    index = np.argmin(J_history)
    c = Label_history[index]
    centroids = C_history[index]
    return cost, index, c, centroids