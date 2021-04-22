
def search_images_ddg_corrected(term, max_images=200):
    "Search for `term` with DuckDuckGo and return a unique urls of about `max_images` images"
    assert max_images<1000
    url = 'https://duckduckgo.com/'
    res = urlread(url,data={'q':term})
    searchObj = re.search(r'vqd=([\d-]+)\&', res)
    assert searchObj
    requestUrl = url + 'i.js'
    params = dict(l='us-en', o='json', q=term, vqd=searchObj.group(1), f=',,,', p='1', v7exp='a')
    urls,data = set(),{'next':1}
    while len(urls)<max_images and 'next' in data:
        try:
            data = urljson(requestUrl,data=params)
            urls.update(L(data['results']).itemgot('image'))
            requestUrl = url + data['next']
        except (URLError,HTTPError): pass
        time.sleep(0.2)
    return L(urls)

urls1 = search_images_ddg_corrected('Sachin Tendulkar', max_images=100)
urls2 = search_images_ddg_corrected('Harbhajan Singh', max_images=100)
for i in range(100):
    name1="Sachin"+str(i)+".jpg"
    download_url(urls1[i], name1)
    name2="Harbhajan"+str(i)+".jpg"
    download_url(urls1[i], name2)
    
