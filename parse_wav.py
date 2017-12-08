import requests
import httplib2
from bs4 import BeautifulSoup,SoupStrainer
import time
import urllib2

#url = "https://tspace.library.utoronto.ca/handle/1807/24487/browse?type=title&submit_browse=Title"
url = "https://tspace.library.utoronto.ca/handle/1807/24501"
req = urllib2.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'})
html = urllib2.urlopen(req).read()



def getURL(page):
    """

    :param page: html of web page (here: Python home page) 
    :return: urls in that page 
    """
    start_link = page.find("a href")
    if start_link == -1:
        return None, 0
    start_quote = page.find('"', start_link)
    end_quote = page.find('"', start_quote + 1)
    url = page[start_quote + 1: end_quote]
    return url, end_quote




if __name__ == '__main__':

    urls = ["1","2"]
    parent_url = ""
    # for url in urls[:1]:
    #     http = httplib2.Http()
    #     complete_url = parent_url + url[1:]
    #     complete_url = "https://tspace.library.utoronto.ca/handle/1807/24501"
    #     req = urllib2.Request(complete_url, None, headers)
    #     html = urllib2.urlopen(req).read()
    #     for link in BeautifulSoup(req, "lxml", parseOnlyThese=SoupStrainer('a')):
    #         if link.has_attr('href'):
    #             print link['href']
    #     print all_urls
        #print page
