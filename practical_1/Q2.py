import pandas as pd

if __name__ == "__main__":
    data_a = pd.read_csv('specs/AutoMpg_question2_a.csv')
    data_b = pd.read_csv('specs/AutoMpg_question2_b.csv')
    data_b = data_b.rename(columns={'name': 'car name'})
    data_a['other'] = 1
    data_c = pd.concat((data_a, data_b))
    data_c.to_csv("./output/question2_out.csv")
