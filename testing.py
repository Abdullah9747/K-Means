from sklearn import datasets
import random
import numpy as np

dataset=datasets.load_iris()


x=dataset.data



class KMeans:
    def __init__(self,n_clusters):
        self.n_clusters=n_clusters
    def init_centroids(self,x):
        return random.sample(list(x),self.n_clusters)
    def Euc_distance(self,a,b):
        return np.sqrt(np.sum((a-b)**2))
    def fit(self,x,n_iter=10):
        centroids=self.init_centroids(x)
        for _ in range(n_iter):
            clusters={i:[] for i in range(self.n_clusters)}
            for point in x:
                distances=[self.Euc_distance(point,centroid) for centroid in centroids]
                cluster_assigned=np.argmin(distances)
                clusters[cluster_assigned].append(point)
            for i in range(self.n_clusters):
                centroids[i]=np.mean(clusters[i],axis=0)
        self.centroids=centroids
        return centroids,clusters

kmeans=KMeans(3)
centroids,clusters=kmeans.fit(x,50)
print(centroids)
        
        


