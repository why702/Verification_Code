# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:53:35 2017

@author: HsiaoYuh_Wang
"""
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image

import CodeSegment_32
import CodeResult

driver = webdriver.Firefox()
driver.get("http://railway.hinet.net/ctno1.htm")

person_id = driver.find_element_by_id("person_id")
person_id.send_keys("F126358632")

from_station = driver.find_element_by_id('from_station')
for option in from_station.find_elements_by_tag_name('option'):
    if option.text.find('100-台北') != -1:
        option.click()
        break
        
to_station = driver.find_element_by_id('to_station')
for option in to_station.find_elements_by_tag_name('option'):
    if option.text.find('051-花蓮') != -1:
        option.click()
        break
        
getin_date = driver.find_element_by_id('getin_date')
for option in getin_date.find_elements_by_tag_name('option'):
    if option.text.find('2017/12/30【六】') != -1:
        option.click()
        break
        
train_no=driver.find_element_by_id("train_no")
train_no.send_keys("248")

action = webdriver.common.action_chains.ActionChains(driver)
action.move_to_element_with_offset(train_no, 100, 0)
action.click()
action.perform()

order_qty_str = driver.find_element_by_id('order_qty_str')
for option in order_qty_str.find_elements_by_tag_name('option'):
    if option.text == '6':
        option.click()
        break
    
n_order_qty_str = driver.find_element_by_id('n_order_qty_str')
for option in n_order_qty_str.find_elements_by_tag_name('option'):
    if option.text == '6':
        option.click()
        break
    
train_no.send_keys(Keys.RETURN)
time.sleep(1)
        
driver.save_screenshot('./test.png')


idRandomPic = driver.find_element_by_id('idRandomPic')

left = idRandomPic.location['x']
right = idRandomPic.location['x'] + idRandomPic.size['width']
top = idRandomPic.location['y']
bottom = idRandomPic.location['y'] + idRandomPic.size['height']

imagePath = 'test.jpg'

img = Image.open('./test.png')
img = img.crop((left, top, right, bottom))

#img.show()
if len(img.split()) == 4:
   r,g,b,a=img.split()
   toImage=Image.merge("RGB",(r,g,b))
   toImage.save(imagePath)


N_CLASSES = 10
IMG_W = 32
IMG_H = 32
CHANNEL = 3
#BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 20000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate = 0.0001
dropout = 1
    
listROI = CodeSegment_32.segmentCode(imagePath);
resultList, possibilityList = CodeResult.evaluate_images(listROI)

result = ''
for s in resultList:
    result = result + str(s)
    

idRandomPic1 = driver.find_element_by_id('idRandomPic')
idRandomPic1.send_keys(Keys.TAB)
idRandomPic1.send_keys(Keys.TAB)
idRandomPic1.send_keys(result)
#idRandomPic1.send_keys(Keys.RETURN)

#driver.close()