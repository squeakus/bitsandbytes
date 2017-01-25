import re
from BeautifulSoup import BeautifulSoup

doc = '<table class=\'data\'><tr class=\'header\'><td colspan=\'2\'><a href=\'http://www.irlweather.com/\' title=\'http://www.irlweather.com/\'>Killucan</a></td><td class=\'r\'>2012-11-24 16:24:41</td></tr><tr class=\'temp_row\'><td><img src=\'images/conditions/11.gif\'/>&nbsp;</td><td><span class=\'temp\'>3.1&deg;C</span></td><td class=\'r\'><span class=\'temp_high\'>High:</span> 3.1&deg;C<br/><span class=\'temp_low\'>Low:</span> -2.3&deg;C</td></tr><tr class=\'wind_row\'><td><img src=\'images/wind/SW.png\'/>&nbsp;</td><td>Wind: 2 km/h</td><td class=\'r\'>Gust: 1.7 km/h</td></tr><tr class=\'data_row\'><td colspan=\'2\'>Dew Point: 2.5&deg;C</td><td class=\'r\'>Humidity: 96%</td></tr><tr class=\'data_row\'><td colspan=\'2\'>Pressure: 1006.2 mb <img title=\'Falling Slowly\' src=\'images/fallings.gif\'></td><td class=\'r\'>Rain Today: 0.2 mm</td></tr><tr class=\'data_row\'><td style=\'text-align: center;\' colspan=\'3\'><span style=\'font-size: 8px\'></span></td></tr></table>'

soup = BeautifulSoup(doc)
#print soup.prettify()

info = soup.findAll('tr',{'class':'wind_row'})
for data in info:
    direction = str(data.find('img'))
    direction = direction.rstrip('.png" />')
    direction = direction.lstrip('<img src="images/wind/')
    print direction
    m = re.search('Wind:(.+?)km', str(data))
    if m:
        speed = m.group(1)
        print speed

