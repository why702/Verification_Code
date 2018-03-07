import cv2
import requests
import PIL
import numpy
from PIL import Image
from matplotlib import pyplot as plt

IMG_SIZE = 32 
TEST_SIZE_W = 200
TEST_SIZE_H = 60

for i in range(100):
    
    with open('test.jpg', 'wb') as fig:
        res = requests.get('http://railway.hinet.net/ImageOut.jsp?pageRandom=0.9013912568365248')
        fig.write(res.content)
    
    
    #image = Image.open('test.jpg')
    pil_img = PIL.Image.open('test.jpg').convert('RGB')
    opencv_img = numpy.array(pil_img)
    debug_img = numpy.array(pil_img)
    
    imgray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite('test0.jpg', thresh)
    
    #filter the vertical and horizonal lines
    width = thresh.shape[1]
    height = thresh.shape[0]
    for x in range(1, width - 1):
        for y in range(1, height - 1):
           p00 = thresh[y - 1   , x - 1]
           p10 = thresh[y       , x - 1]
           p20 = thresh[y + 1   , x - 1]
           p01 = thresh[y - 1   , x]
           p11 = thresh[y       , x]
           p21 = thresh[y + 1   , x]
           p02 = thresh[y - 1   , x + 1]
           p12 = thresh[y       , x + 1]
           p22 = thresh[y + 1   , x + 1]
    
           if p00 == 255 and p10 == 255 and p20 == 255 and p01 == 255 and p11 == 0 and p21 == 0 and p02 == 255 and p12 == 255 and p22 == 255:
               thresh[y, x] = 255
           elif p00 == 255 and p10 == 255 and p20 == 255 and p01 == 255 and p11 == 0 and p21 == 255 and p02 == 255 and p12 == 0 and p22 == 255:
               thresh[y, x] = 255
           elif p00 == 255 and p10 == 255 and p20 == 255 and p01 == 0 and p11 == 0 and p21 == 255 and p02 == 255 and p12 == 255 and p22 == 255:
               thresh[y, x] = 255
           elif p00 == 255 and p10 == 0 and p20 == 255 and p01 == 255 and p11 == 0 and p21 == 255 and p02 == 255 and p12 == 255 and p22 == 255:
               thresh[y, x] = 255
           elif p00 == 255 and p10 == 255 and p20 == 255 and p01 == 255 and p11 == 0 and p21 == 0 and p02 == 255 and p12 == 0 and p22 == 255:
               thresh[y, x] = 255
               thresh[y + 1, x] = 255
               thresh[y, x + 1] = 255
           elif p00 == 255 and p10 == 0 and p20 == 255 and p01 == 0 and p11 == 0 and p21 == 0 and p02 == 255 and p12 == 0 and p22 == 255:
               thresh[y, x] = 255
               thresh[y + 1, x] = 255
               thresh[y, x + 1] = 255
               thresh[y - 1, x] = 255
               thresh[y, x - 1] = 255
           elif p00 == 255 and p10 == 0 and p20 == 255 and p01 == 255 and p11 == 0 and p21 == 0 and p02 == 255 and p12 == 0 and p22 == 255:
               thresh[y, x] = 255
               thresh[y + 1, x] = 255
               thresh[y, x + 1] = 255
               thresh[y - 1, x] = 255
               thresh[y, x - 1] = 255
           elif p00 == 255 and p10 == 255 and p20 == 255 and p01 == 0 and p11 == 0 and p21 == 0 and p02 == 255 and p12 == 0 and p22 == 255:
               thresh[y, x] = 255
               thresh[y + 1, x] = 255
               thresh[y, x + 1] = 255
               thresh[y - 1, x] = 255
               thresh[y, x - 1] = 255
           elif p00 == 255 and p10 == 255 and p20 == 255 and p01 == 0 and p11 == 0 and p21 == 0 and p02 == 255 and p12 == 255 and p22 == 255:
               thresh[y, x] = 255
               thresh[y + 1, x] = 255
               thresh[y, x + 1] = 255
               thresh[y - 1, x] = 255
               thresh[y, x - 1] = 255
           elif p00 == 255 and p10 == 0 and p20 == 255 and p01 == 255 and p11 == 0 and p21 == 255 and p02 == 255 and p12 == 0 and p22 == 255:
               thresh[y, x] = 255
               thresh[y + 1, x] = 255
               thresh[y, x + 1] = 255
               thresh[y - 1, x] = 255
               thresh[y, x - 1] = 255
           elif p00 == 0 and p10 == 255 and p20 == 0 and p01 == 0 and p11 == 255 and p21 == 0 and p02 == 0 and p12 == 255 and p22 == 0:
               thresh[y - 1   , x] = 0
               thresh[y       , x - 1] = 0
               thresh[y       , x + 1] = 0
           elif p00 == 0 and p10 == 0 and p20 == 0 and p01 == 255 and p11 == 255 and p21 == 255 and p02 == 0 and p12 == 0 and p22 == 0:
               thresh[y - 1   , x] = 0
               thresh[y       , x] = 0
               thresh[y + 1   , x] = 0
    
           if x < (width - 3) and y < (height - 3):
               #p00 = thresh[y - 1 , x - 1]
               #p10 = thresh[y , x - 1]
               #p20 = thresh[y + 1 , x - 1]
               p30 = thresh[y + 2   , x - 1]
               #p01 = thresh[y - 1 , x]
               #p11 = thresh[y , x]
               #p21 = thresh[y + 1 , x]
               p31 = thresh[y + 2   , x]
               #p02 = thresh[y - 1 , x + 1]
               #p12 = thresh[y , x + 1]
               #p22 = thresh[y + 1 , x + 1]
               p32 = thresh[y + 2   , x + 1]
               p03 = thresh[y - 1   , x + 2]
               p13 = thresh[y       , x + 2]
               p23 = thresh[y + 1   , x + 2]
               p33 = thresh[y + 2   , x + 2]
    
               if p00 == 255 and p10 == 0 and p20 == 0 and p30 == 255 and p01 == 255 and p11 == 0 and p21 == 0 and p31 == 255 and p02 == 255 and p12 == 0 and p22 == 0 and p32 == 255:
                   thresh[y       , x - 1] = 255
                   thresh[y + 1   , x - 1] = 255
                   thresh[y       , x] = 255
                   thresh[y + 1   , x] = 255
                   thresh[y       , x + 1] = 255
                   thresh[y + 1   , x + 1] = 255
               elif p00 == 255 and p10 == 255 and p20 == 255 and p01 == 0 and p11 == 0 and p21 == 0 and p02 == 0 and p12 == 0 and p22 == 0 and p03 == 255 and p13 == 255 and p23 == 255:
                   thresh[y - 1   , x] = 255
                   thresh[y       , x] = 255
                   thresh[y + 1   , x] = 255
                   thresh[y - 1   , x + 1] = 255
                   thresh[y       , x + 1] = 255
                   thresh[y + 1   , x + 1] = 255
    
    
    
    
    cv2.imwrite('test_vh.jpg', thresh)
    
    image, contours, hieraichy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    img1 = cv2.imread('test.jpg')
    cv2.drawContours(img1, contours, -1, (0,255,0), 1)
    #cv2.imwrite('test_all.jpg', img1)
    
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key = lambda x:x[1])
    
    ary = []
    for (c,_) in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        #print((x,y,w,h))
        if y < 1 or y > 51:
            continue
        elif w > 5 and w < 100 and h > 5 and h < 100:
            ary.append((x,y,w,h))
            cv2.rectangle(debug_img,(x,y),(x + w,y + h),(0,0,255))
            
    ary = sorted(ary, key = lambda x: x[0]**2 + x[1]**2)
    print(ary)
    
    bIgnore = False
    i = 0
    while i < len(ary):
        
        (x0,y0,w0,h0) = ary[i]
        print('ary[i] = {}'.format(ary[i]))
    
        if i == len(ary) - 1:
            (x1,y1,w1,h1) = ary[i - 1]
            print('ary[i - 1] = {}'.format(ary[i - 1]))
        else:
            (x1,y1,w1,h1) = ary[i + 1]
            print('ary[i + 1] = {}'.format(ary[i + 1]))
        
        iCloseDistance = 3
        x_0 = x0 + w0 + iCloseDistance
        y_0 = y0 + h0 + iCloseDistance
        x_1 = x1 + w1 + iCloseDistance
        y_1 = y1 + h1 + iCloseDistance
        
        #two ROI are closed        
        if x_0 >= x1 and x0 <= x_1 and y_0 >= y1 and y0 <= y_1:
            
            #two ROI both big
            if w0 + h0 >= 32 and w1 + h1 >= 32:
                ary[i] = (x0,y0,w0,h0)
                i += 1
                continue
    
            overlapX = min(x_0,x_1) - max(x0,x1)
            overlapY = min(y_0,y_1) - max(y0,y1)
            print('overlapX = %d' %overlapX)
            print('overlapY = %d' %overlapY)  
                
            '''
            if overlapX < 4 and overlapY < 4:
                #overlap too small            
            else:
                #overlap too much      
            '''    
    
            #to avoid the ROI become too big after merged.
            if (max(x_0,x_1) - min(x0,x1)) >= IMG_SIZE and (max(y_0,y_1) - min(y0,y1)) >= IMG_SIZE:
                if overlapX + overlapY > 13:
                    print('too close, merge!')
                else:
                    ary[i] = (x0,y0,w0,h0)
                    i += 1
                    continue
            
            #merge
            x0 = min(x0,x1)
            w0 = max(x_0,x_1) - x0
            y0 = min(y0,y1)
            h0 = max(y_0,y_1) - y0
            if i == len(ary) - 1:
                ary[i] = (x0,y0,w0,h0)
                del ary[i-1]
            else:
                ary[i] = (x0,y0,w0,h0)
                del ary[i+1]
            bIgnore = True
    
        else:
            iboundary = 15
            if w0 < iboundary:
                dif = iboundary - w0
                x0 -= int (dif / 2)
                if x0 < 0:
                    x0 = 0
                w0 = iboundary
            if h0 < iboundary:
                dif = iboundary - h0
                y0 -= int (dif / 2)
                if y0 < 0:
                    y0 = 0
                h0 = iboundary
            ary[i] = (x0,y0,w0,h0)        
        
    
        if bIgnore == False:
            i += 1
        else:
            bIgnore = False
    
    #remove too smail ROI
    i = 0
    while i < len(ary):
        print(ary[i])
        (x0,y0,w0,h0) = ary[i]
        if w0 + h0 < 20:
            del ary[i]
            continue
        elif h0 > IMG_SIZE:
            y_0 = y0 + h0
            if y_0 > 51:
                ary[i] = (x0, y0, w0, IMG_SIZE)
            else:
                y0_0 = int(y0 - (h0 - w0) / 2)
                h0_0 = IMG_SIZE
                ary[i] = (x0, y0_0, w0, h0_0)
        
        (x0,y0,w0,h0) = ary[i]
            
        if w0 >= 29 :
            w0_0 = int (w0 / 2)
            ary[i] = (x0, y0, w0_0, h0)
            
            x0_1 = x0 + w0_0
            w0_1 = w0 - w0_0
            ary.insert(i + 1, [x0_1, y0, w0_1, h0])
        i += 1
        
    #let ROI be IMG_SIZE x IMG_SIZE
    i = 0
    while i < len(ary):
        print(ary[i])
        (x0,y0,w0,h0) = ary[i]
        
        if w0 < IMG_SIZE:
            x0 = int (x0 - (IMG_SIZE - w0) / 2)
        else:
            x0 = int (x0 + (w0 - IMG_SIZE) / 2)
        w0 = IMG_SIZE
        
        if h0 < IMG_SIZE:
            y0 = int (y0 - (IMG_SIZE - h0) / 2)
        else:
            y0 = int (y0 + (h0 - IMG_SIZE) / 2)
        h0 = IMG_SIZE
        
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x0 + w0 > TEST_SIZE_W:
            x0 = TEST_SIZE_W - w0
        if y0 + h0 > TEST_SIZE_H:
            y0 = TEST_SIZE_H - h0
        
        
        ary[i] = (x0, y0, w0, h0)        
        cv2.rectangle(debug_img,(x0,y0),(x0 + w0,y0 + h0),(255,0,0))
        i += 1
        
    print('ary1 = {}'.format(ary))
    cv2.imwrite('test_filter.jpg', debug_img)
    
    plt.imshow(debug_img)
    plt.show()
    
    import uuid
    
    for id, (x,y,w,h) in enumerate(ary):
        roi = opencv_img[y:y + h, x:x + w]
        ROI = roi.copy()
        ranStr = str(uuid.uuid4())
        plt.imshow(ROI)
        plt.show()
        outfile0 = './database0/%s.jpg' % (ranStr)
        cv2.imwrite(outfile0, ROI)    