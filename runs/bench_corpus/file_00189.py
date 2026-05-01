from PycharmProjects.PythonSeleniumAutoAugSep.browserLaunch import launchChrome as op

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

op.driver.get("https://www.amazon.in")
op.driver.fullscreen_window()
search =op.driver.find_element_by_id("twotabsearchtextbox")
search.send_keys("ipad pro",Keys.ENTER)

time.sleep(2)
search = op.driver.find_element(By.ID,"twotabsearchtextbox")
time.sleep(2)
search.clear()
search.send_keys("ipad pro keyboard",Keys.ENTER)
time.sleep(3)

op.driver.close()