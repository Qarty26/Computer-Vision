import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 1.2

print(cv.__version__)

# 1.3

image  = cv.imread("butterfly.jpeg")
print(image)

# 1.4

image  = cv.imread("butterfly.jpeg", cv.IMREAD_GRAYSCALE)
print(image)

# 1.5

image  = cv.imread("butterfly.jpeg")
cv.imshow('butterfly', image)
cv.waitKey(0)
cv.destroyAllWindows()

# 1.6

img = cv.resize(cv.cvtColor(cv.imread("football.jpg"), cv.COLOR_BGR2GRAY),(100, 100))

#a

sorted_img = np.sort(img, axis = None)
col_img = sorted_img.reshape(10000,1)
plt.plot(col_img)
plt.show()


#b

drjos = img[50:100, 50:100]
cv.imshow("",drjos)
cv.waitKey(0)
cv.destroyAllWindows()

#or

plt.imshow(drjos, cmap="gray")
plt.show()

#c

t = img.mean()
print(t)

#d

B = np.where(img >= t, 255, 0).astype(np.uint8)
cv.imshow("",B)
cv.waitKey(0)
cv.destroyAllWindows()

#e
C1 = img - t
C1[C1 < 0] = 0  
C1 = C1.astype(np.uint8) 
cv.imshow("",C1)
cv.waitKey(0)
cv.destroyAllWindows()

#  or

C = np.where(img-t >=0, img-t,0).astype(np.uint8)
cv.imshow("",C)
cv.waitKey(0)
cv.destroyAllWindows()



#f

minn = np.min(img)

poz = np.where(img == minn)
for coord in zip(poz[0], poz[1]):
    print(coord)



#1.7


def load_img(folder):
    images = []
    for img_name in os.listdir(folder):  
        img = cv.imread(f'{folder}/{img_name}')  
        if img is not None:  
            images.append(img)
    return images



def conv_albnegru(images):
    grayscale_images = []
    

    for img in images:

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grayscale_images.append(gray_img)
    
    return grayscale_images


S1 = load_img("colectiiImagini/set1")
S2 = load_img("colectiiImagini/set2")


S11 = conv_albnegru(S1)
S21 = conv_albnegru(S2)

def calc_medie(imgs):

    if len(imgs) == 0:
        return None


    if len(imgs[0].shape) == 3: 

        h, w, c = imgs[0].shape
        
        avg_img = np.zeros((h, w, c), np.float32)
        
        for img in imgs:
            avg_img += img.astype(np.float32)
        
        avg_img /= len(imgs)
        
        avg_img = avg_img.astype(np.uint8)
        
    elif len(imgs[0].shape) == 2: 

        h, w = imgs[0].shape

        avg_img = np.zeros((h, w), np.float32)
        
        for img in imgs:
            avg_img += img.astype(np.float32)

        avg_img /= len(imgs)
        

        avg_img = avg_img.astype(np.uint8)
        
 
    return avg_img 


img_medie = calc_medie(S21)

if img_medie is not None:
    cv.imshow('Imaginea Medie', img_medie)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def calculate_std_deviation(images):

    if len(images) == 0:
        return None

    h, w = images[0].shape
    
    
    sum_img = np.zeros((h, w), np.float32)
    sum_img_sq = np.zeros((h, w), np.float32)


    for img in images:
        sum_img += img.astype(np.float32)
        sum_img_sq += img.astype(np.float32) ** 2


    N = len(images)

    mean_img = sum_img / N
    std_dev_img = np.sqrt((sum_img_sq / N) - (mean_img ** 2))
    std_dev_img = np.clip(std_dev_img, 0, 255).astype(np.uint8)
    
    return std_dev_img

img2 = calculate_std_deviation(S21)
cv.imshow('Deviatie', img2)
cv.waitKey(0)
cv.destroyAllWindows()


# 1.8

def extract_500samples(image_name):
    L = []
    img = cv.imread(image_name)
    h, w, c  = img.shape
    
    for _ in range(10):
        x = random.randint(0, h-20)
        y = random.randint(0, w-20)
        L.append(img[x:x+20, y:y+20])
        
    return L, img

def l2_distance(img1, img2):
    return np.sqrt(np.sum((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))
    
def replace_closest(image_name):
    
    L, img = extract_500samples(image_name)
    sample = img[250:270, 250:270]
    temp = sample
    
    minn = 1e9
    for i in L:
        dist = l2_distance(i, sample)
        if dist < minn:
            dist = minn
            temp = i
    
    
    img[250:270, 250:270] = temp
    cv.imwrite("butterfly_modified.jpeg", img)
    
    
replace_closest("butterfly.jpeg")