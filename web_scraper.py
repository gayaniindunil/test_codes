import requests 
from bs4 import BeautifulSoup

result = requests.get("http://www.google.com/")

# print(result.status_code)
# print(result.headers)


src = result.content
# print(src) #print the source of the web page


# soup = BeautifulSoup(src,"lxml")
soup = BeautifulSoup(src)

links = soup.find_all('a')
# print(links)

for link in links:
    if "About" in link.text:
        print(link)
        print(link.attrs['href'])

