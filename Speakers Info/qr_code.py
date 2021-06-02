import pandas as pd
import qrcode

def conversion(text):
    return text.title()
    

df = pd.read_csv('ravdess_support.csv')
df['Name'] = df['Name'].map(conversion)

print(df['Name'].head())

df.to_csv('25_actors.csv', index= False)