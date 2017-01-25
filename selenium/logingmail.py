from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox()

driver.get("http://www.gmail.com")
element = driver.find_element_by_id("Email")
element.send_keys("jonathanbyrn@gmail.com")

driver.implicitly_wait(10)

element1 = driver.find_element_by_id("Passwd")
element1.send_keys("fuck1254")
element.submit()

driver.implicitly_wait(5)

print "moo"
driver.find_element_by_css_selector("span.gb_4.gbii").click()
driver.find_element_by_id("gb_71").click()
#print driver.page_source
driver.close()
