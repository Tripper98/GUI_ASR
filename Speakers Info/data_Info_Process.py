import pandas as pd 
import qrcode 


# Id Name Image More

df_test = pd.read_csv('Speakers Info\\100_speakers.csv')

man_link = 'https://user-images.githubusercontent.com/46791116/121378219-e871f280-c93a-11eb-9b3e-3aa9e5d44c26.png'
woman_link = 'https://user-images.githubusercontent.com/46791116/121378211-e740c580-c93a-11eb-8113-82bf99107219.png'

df_test['Image'] = df_test['Id'].apply(lambda x : man_link if x<50 else woman_link)

df_test.to_csv('Speakers Info\\youtube_speakers.csv', index= False)
