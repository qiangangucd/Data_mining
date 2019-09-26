import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('specs/AutoMpg_question1.csv')
    print(data.isnull().sum())
    '''
    horsepower      16
    mpg              0
    cylinders        0
    displacement     0
    weight           0
    acceleration     0
    model year       0
    origin          15
    car name         0
    '''
    data.horsepower = data.horsepower.fillna(data.horsepower.mean())
    data.origin = data.origin.fillna(data.origin.min())
    data.to_csv("./output/question1_out.csv")
