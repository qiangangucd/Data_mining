import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import preprocessing
import pydotplus

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/marks_question1.csv'))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color='r')
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'output/marks.png'))
plt.savefig(os.path.join(os.path.dirname(__file__), 'output/marks.png'))
plt.show()

x = pd.DataFrame(data.midterm)
y = pd.DataFrame(data.final)
# theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
# print(theta.dot(x.T)-y.T)


reg = LinearRegression().fit(x, y)
print(reg.predict(np.array(86).reshape(-1, 1)))
data3 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/borrower_question2.csv'))
data2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/borrower_question2.csv')).drop(columns='TID')
# data2 = data_2.drop(columns='TID')
le = preprocessing.LabelEncoder()
for c in range(data2.shape[1]):
    if c != 2:
        data2.iloc[:, c] = le.fit_transform(data2.iloc[:, c])
clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.5, criterion='entropy')
clf.fit(data2.iloc[:, :-1], data2.iloc[:, -1])
tree.plot_tree(clf.fit(data2.iloc[:, :-1], data2.iloc[:, -1]))

# Load libraries
from IPython.display import Image

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=data2.columns[:-1],
                                class_names=data2.columns[-1])

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
# Image(graph.create_png())
# Create PNG
graph.write_png("borrower.png")
