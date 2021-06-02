from logging import INFO
import pandas as pd

Actors_me = pd.read_csv("Speakers Info\\ravdess_support.csv", index_col = False)
# print(Actors_me.head())
# x = Actors_me.loc[Actors_me.Id == 6]
# print(x['Name'][0])
info_speaker = Actors_me.loc[Actors_me['Id']== 6]
print(info_speaker['Image'].values)