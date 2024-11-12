import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
iris = load_iris()
X = iris.data
feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def pam_clustering(X, k, max_iters=100):
    n_samples = X.shape[0]
    
    # 随机初始化中心点
    medoid_indices = np.random.choice(n_samples, k, replace=False)
    medoids = X[medoid_indices]
    
    for _ in range(max_iters):
        # 分配样本到最近的中心点
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sum((X - medoids[i])**2, axis=1)
        labels = np.argmin(distances, axis=1)
        
        # 更新每个簇的中心点
        old_medoids = medoids.copy()
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # 计算簇内所有点对之间的总距离
                cluster_distances = np.zeros(len(cluster_points))
                for j in range(len(cluster_points)):
                    cluster_distances[j] = np.sum(
                        np.sum((cluster_points - cluster_points[j])**2, axis=1)
                    )
                # 选择总距离最小的点作为新的中心点
                new_medoid_idx = np.argmin(cluster_distances)
                medoids[i] = cluster_points[new_medoid_idx]
        
        # 检查收敛
        if np.all(old_medoids == medoids):
            break
            
    return labels, medoids

# 使用PAM进行聚类
pam_labels, pam_medoids = pam_clustering(X_scaled, k=3)

# 创建可视化函数
def plot_clusters(X, labels, medoids, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='*', 
                s=200, label='中心点')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title, fontsize=12)
    plt.colorbar(scatter)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# 绘制聚类结果
plot_clusters(X_scaled, pam_labels, pam_medoids, 'PAM聚类结果')

# 输出聚类结果统计
def print_cluster_stats(labels):
    unique_labels = np.unique(labels)
    print("\nPAM聚类结果：")
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"类别 {label}: {count}个样本")

print_cluster_stats(pam_labels)

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 获取DBSCAN聚类中心点（每个簇的质心）
dbscan_centers = []
unique_labels = np.unique(dbscan_labels)
for label in unique_labels:
    if label != -1:  # 排除噪声点（标签为-1）
        mask = dbscan_labels == label
        cluster_points = X_scaled[mask]
        center = np.mean(cluster_points, axis=0)
        dbscan_centers.append(center)
dbscan_centers = np.array(dbscan_centers)

# 绘制DBSCAN聚类结果
plt.figure(figsize=(10, 8))
# 绘制聚类点
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                     c=dbscan_labels, cmap='viridis')
# 绘制聚类中心
if len(dbscan_centers) > 0:
    plt.scatter(dbscan_centers[:, 0], dbscan_centers[:, 1], 
                c='red', marker='*', s=200, label='聚类中心')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('DBSCAN聚类结果', fontsize=12)
plt.colorbar(scatter)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# 输出DBSCAN聚类结果统计
print("\nDBSCAN聚类结果：")
for label in unique_labels:
    count = np.sum(dbscan_labels == label)
    if label == -1:
        print(f"噪声点: {count}个样本")
    else:
        print(f"类别 {label}: {count}个样本")
