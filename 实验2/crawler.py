
import requests
import re
response = requests.get('http://www.hit.edu.cn')

from bs4 import BeautifulSoup

soup = BeautifulSoup(response.content,'html.parser')
#print(soup.prettify())

import json
dic = {}
for link in soup.find_all('a'):
#    print(link.get('href'),link.get('title'))
    dic[link.get('title')] = link.get('href')
json_str = json.dumps(dic,ensure_ascii=False)
print(json_str)