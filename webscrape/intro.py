from bs4 import BeautifulSoup as soup 
# from urllib.request import urlopen as uReq
import requests

my_url = 'http://thessaloniki.travel/el/exerevnontas-tin-poli/endiaferouses-geitonies'

# Open the connection
uClient = requests.get(my_url)

# Read the page and store it to a variable
page_html = uClient.read()

# Close the connection
uClient.close()

# Transform the html into a soup object
page_soup = soup(page_html, "html.parser")

geitonies = page_soup.findAll("div", {"class":"sub_category_info"})

print("Number of geitonies is: ", len(geitonies))
print(geitonies[0])