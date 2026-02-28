import pandas as pd


def feature_engg(data: pd.Dataframe):
    corr = data.corr

    print(corr)


    