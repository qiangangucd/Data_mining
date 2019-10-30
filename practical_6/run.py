import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Question 1
data_q1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/question_1.csv'))
kmeans_q1 = KMeans(n_clusters=3, random_state=0).fit(data_q1)
data_q1['cluster'] = kmeans_q1.predict(data_q1)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    data_q1.to_csv(os.path.join(os.path.dirname(__file__), 'output/question_1.csv'), index=False)
data_q1.to_csv(os.path.join(os.path.dirname(__file__), 'output/question_1.csv'), index=False)

# Question 2
data_q = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/question_2.csv'))
data_q2 = data_q.drop(columns=['NAME', 'MANUF', 'TYPE', 'RATING'])
kmeans_q2_1 = KMeans(n_clusters=5, n_init=5, max_iter=100, random_state=0).fit(data_q2)
data_q2['config1'] = kmeans_q2_1.predict(data_q2)
kmeans_q2_2 = KMeans(n_clusters=5, n_init=100, max_iter=100, random_state=0).fit(data_q2.iloc[:, :-1])
data_q2['config2'] = kmeans_q2_2.predict(data_q2.iloc[:, :-1])

kmeans_q2_3 = KMeans(n_clusters=3, random_state=0).fit(data_q2.iloc[:, :-2])
data_q2['config3'] = kmeans_q2_3.predict((data_q2.iloc[:, :-2]))

data_q[data_q2.columns[-3]] = data_q2.iloc[:, -3]
data_q[data_q2.columns[-2]] = data_q2.iloc[:, -2]
data_q[data_q2.columns[-1]] = data_q2.iloc[:, -1]

data_q.to_csv(os.path.join(os.path.dirname(__file__), 'output/question_2.csv'), index=False)

# Question 3
data_q3 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/question_3.csv'))
data_q3_d = data_q3.drop(columns='ID')
kmeans_q3_1 = KMeans(n_clusters=7, n_init=5, max_iter=100, random_state=0).fit(data_q3_d)
data_q3_d['kmeans'] = kmeans_q3_1.predict(data_q3_d)
plt.scatter(data_q3_d.iloc[:, 0], data_q3_d.iloc[:, 1], c=data_q3_d.iloc[:, -1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cluster')
plt.savefig(os.path.join(os.path.dirname(__file__), 'output/question_3_1.pdf'), bbox_inches='tight')
plt.show()

scaler = MinMaxScaler()
data_q3_d.iloc[:, :-1] = scaler.fit_transform(data_q3_d.iloc[:, :-1])
clustering = DBSCAN(eps=0.04, min_samples=4).fit(data_q3_d.iloc[:, :-1])
data_q3_d['dbscan1'] = clustering.fit_predict(data_q3_d.iloc[:, :-1])
plt.scatter(data_q3_d.iloc[:, 0], data_q3_d.iloc[:, 1], c=data_q3_d.iloc[:, -1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('dbscan1')
plt.savefig(os.path.join(os.path.dirname(__file__), 'output/question_3_2.pdf'), bbox_inches='tight')
plt.show()

clustering = DBSCAN(eps=0.08, min_samples=4).fit(data_q3_d.iloc[:, :-2])
data_q3_d['dbscan2'] = clustering.fit_predict(data_q3_d.iloc[:, :-2])
plt.scatter(data_q3_d.iloc[:, 0], data_q3_d.iloc[:, 1], c=data_q3_d.iloc[:, -1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('dbscan2')
plt.savefig(os.path.join(os.path.dirname(__file__), 'output/question_3_3.pdf'), bbox_inches='tight')
plt.show()

data_q3_d.to_csv(os.path.join(os.path.dirname(__file__), 'output/question_3.csv'), index=False)
