import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Question 1
data_gpa = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/gpa_question1.csv'))
dataset_gpa = data_gpa.drop(columns='count')

data_ohe = pd.get_dummies(dataset_gpa)
frequent_itemsets = apriori(data_ohe, use_colnames=True, min_support=0.15)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    frequent_itemsets.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out_apriori.csv'), index=False)
frequent_itemsets.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out_apriori.csv'), index=False)

rules9_gpa = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    rules9_gpa.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out_rules9.csv'), index=False)
rules9_gpa.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out_rules9.csv'), index=False)

rules7_gpa = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    rules7_gpa.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out_rules7.csv'), index=False)
rules7_gpa.to_csv(os.path.join(os.path.dirname(__file__), 'output/question1_out_rules7.csv'), index=False)

# Question 2
data_bank = pd.read_csv(os.path.join(os.path.dirname(__file__), 'specs/bank_data_question2.csv'))
dataset_bank = data_bank.drop(columns='id')
list_columns = dataset_bank.columns.tolist()
for c in list_columns:
    if np.issubdtype(dataset_bank[c].dtype, np.number):
        dataset_bank[c] = pd.cut(dataset_bank[c], 3)

bank_ohe = pd.get_dummies(dataset_bank)
frequent_itemsets_bank = fpgrowth(bank_ohe, min_support=0.2, use_colnames=True)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    frequent_itemsets_bank.to_csv(os.path.join(os.path.dirname(__file__), 'output/question2_out_fpgrowth.csv'),
                                  index=False)
frequent_itemsets_bank.to_csv(os.path.join(os.path.dirname(__file__), 'output/question2_out_fpgrowth.csv'), index=False)

rules_bank = association_rules(frequent_itemsets_bank, metric='confidence', min_threshold=0.78)
print(len(rules_bank))
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'output')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'))
    rules_bank.to_csv(os.path.join(os.path.dirname(__file__), 'output/question2_out_rules.csv'),
                                  index=False)
rules_bank.to_csv(os.path.join(os.path.dirname(__file__), 'output/question2_out_rules.csv'), index=False)
