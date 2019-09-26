import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Question 1 Start
# load data
data1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/SensorData_question1.csv'))

# add a new ccolumn
data1['Original Input3'] = round(data1['Input3'], 5)
data1['Original Input12'] = round(data1['Input12'], 5)

# z-score normalisation
data1['Input3'] = (data1['Input3'] - data1['Input3'].mean()) / data1['Input3'].std()

# [0-1] normalisation
data1['Input12'] = (data1['Input12'] - data1['Input12'].min()) / (data1['Input12'].max() - data1['Input12'].min())

# compute the mean of each row
data1['Average Input'] = np.mean(data1, axis=1)

# Check if the 'output' folder exists, create if it does not.
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    data1.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out.csv'), index=False)
data1.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out.csv'), index=False)
# Qestion 1 End


# Question 2 Start
# load data
data2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/DNAData_question2.csv'))
# PCA
pca = PCA(n_components=0.95)
new_data = pca.fit_transform(data2)
# Segment data values into bins of equal width
for idx in range(new_data.shape[1]):
    data2['pca' + str(idx) + '_width'] = pd.cut(new_data[:, idx], 10)
# Segment data values into bins of frequency width
for idx in range(new_data.shape[1]):
    data2['pca' + str(idx) + '_freq'] = pd.qcut(new_data[:, idx], 10)

# Check if the 'output' folder exists, create if it does not.
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    data2.to_csv(os.path.join(os.path.dirname(__file__), 'output/question2_out.csv'), index=False)
data2.to_csv(os.path.join(os.path.dirname(__file__), 'output/question2_out.csv'), index=False)
# Question 2 End
