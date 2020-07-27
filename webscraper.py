import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

url = 'http://www.imdb.com/title/tt'

links_df = pd.read_csv('Data/links.csv')
## Not subtracting movies id by 1, keeping it as given.
links_df['imdbId'] = links_df['imdbId'].apply(lambda x : str(x).zfill(7))
print(links_df.head())

url += links_df.iloc[0]['imdbId']
req = requests.get(url)
soup = BeautifulSoup(req.content, 'html.parser')

## Extracting movie title
div_title = soup.findAll('div', {'class' : 'title_wrapper'})[0]
title_text = div_title.findAll('h1')[0].get_text()
# print(title_text)

## Extracting image posters
div_url = soup.findAll('div', {'class' : 'poster'})[0]
img_url = div_url.findAll('img')[0].get('src')
# print(img_url)

img = Image.open(requests.get(img_url, stream = True).raw)
img.save('Images/{}.jpg'.format(links_df.iloc[0]['movieId']))

## Extracting one liner
div_ol = soup.findAll('div', {'class' : 'summary_text'})[0]
ol_text = div_ol.get_text().strip()
# print(ol_text)

## Extracting director, and main cast
director, cast1, cast2, cast3 = '', '', '', ''
casts = []
for div_item in soup.findAll('div', {'class' : 'credit_summary_item'}):
    if div_item.findAll('h4', {'class' : 'inline'})[0].get_text() == 'Director:':
        director = div_item.findAll('a')[0].get_text().replace(' ', '')
        # print(director)

    for div_cast in div_item.findAll('h4', {'class' : 'inline'}):
        if div_cast.get_text() == 'Stars:':
            casts = [each.get_text().replace(' ', '') for each in div_item.findAll('a')][:3]
            # print(casts)
        else:
            break
cast1, cast2, cast3 = casts

print("Title : {}".format(title_text))
print("One liner : {}".format(ol_text))
print("Director : {}".format(director))
print("Casts : {}, {}, {}".format(cast1, cast2, cast3))
print("------------------------------------------------------------------")
