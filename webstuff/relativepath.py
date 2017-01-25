url = "http://placebo/tutorials/slide1/css"
relpath = "../../images/moo.png"

def relativepath(url, elem):
    count = elem.count('..')
    elem = elem.replace('/..','')
    elem = elem.replace('../','/')
    url = url.split('/')
    url = '/'.join(url[:-count])
    print url + elem

relativepath(url, relpath)
