import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution_bk(img,filt):
    img_n = cv2.normalize(img.astype('float'),0.0,1.0,cv2.NORM_MINMAX) # 데이터를 double type으로 바꿈
    p = int((filt.shape[0] -1)/2)  #패딩 개수 계산
    height =  img.shape[0]
    width = img.shape[1]
    
    img_p = np.zeros((height+p*2,width+p*2))
    img_p[p:-p,p:-p] = img_n  #패딩이 적용된 이미지
    img_c = np.zeros_like(img_n)
    
    for i in range(img_p.shape[0]-(filt.shape[0]-1)):
        for j in range(img_p.shape[1] - (filt.shape[0]-1)):
            mul = img_p[i:i+filt.shape[0],j:j+filt.shape[0]] *filt
            sum_c = sum(sum(mul))
            img_c[i,j] = sum_c
    return img_c


def hist_bk(img): #Make Histogram
    height = img.shape[0]
    width = img.shape[1]
    img_data = np.array(img) # 이미지 pixel 데이터를  array로
    hist_data = np.zeros(256)
    x_g = np.arange(256)
    
    for x in range(height):
        for y in range(width):
            value = img_data[x][y]
            hist_data[value] = hist_data[value] + 1
    return img_data, hist_data


def hist_e_bk(img_data, hist_data):  # Histogram Equalization
    PDF = hist_data /(img_data.shape[0]* img_data.shape[1])
    hist_round = np.zeros(256)
    hist_ru = np.zeros_like(img_data)
    hist_e = np.zeros(256)
    for i in range(256):
        CDF = 0
        for j in range(i+1):
            CDF = CDF + PDF[j]   #누적분포함수 구하기
        hist_round[i] = round(255*CDF)  
    for x in range(img_data.shape[0]):
        for y in range(img_data.shape[1]):
            value1 = img_data[x][y]
            hist_ru[x][y] = hist_round[value1] #원본이미지의 픽셀이 누적분포함수 적용했을 때의 픽셀로 교체
            value2 = hist_ru[x][y] 
            hist_e[value2] = hist_e[value2]+1 #교체한 데이터로 다시 히스토그램 생성
    return hist_e


def BoxF_bk(size): # box filter, 평균 밝기로 픽셀 값 변경
    boxF = np.ones((size,size))/(size**2)
    return boxF


def WeightedMean_bk(): #가중평균필터
    weightedmF = np.array([[1,2,1],[2,4,2],[1,2,1]])
    weightedmF = weightedmF / 16
    return weightedmF


def Gaussian_bk(size,sigma): #가우시안 필터
    Gaussian_filter = np.zeros([size,size])
    for i in range(1,-2, -1):
        for j in range(-1,2):
            Gaussian_filter[i+1,j+1] = np.exp(-(i**2 + j**2)/(2*sigma**2))
    return Gaussian_filter

def MedianF_bk(img,size):
    img_data = np.array(img) # 이미지 pixel 데이터를  array로
    p = int((size -1)/2)  #패딩 개수 계산
    img_p = np.zeros((img_data.shape[0]+p*2,img_data.shape[1]+p*2))
    img_p[p:-p,p:-p] = img_data  #패딩이 적용된 이미지
    img_median = np.zeros((img_p.shape[0],img_p.shape[1]))
    for i in range(0,img_p.shape[0]-(size-1)):
        for j in range(0,img_p.shape[1]-(size-1)):
            a =img_p[i:i+size, j:j+size]
            a = a.reshape(size*size)
            a.sort() #가운데 값을 찾기 위해 정렬
            value = a[int(((size*size)-1)/2)]
            img_median[i][j] =value      
    return img_median

def Laplacian_bk():
    L4 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    L8 = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    return L4, L8

def Unsharpening_bk(img,img_g):
    Unsharp = np.zeros((img.shape[0], img.shape[1]))
    Unsharp = img - img_g
    Unsharpimg = img + Unsharp
    return Unsharp,Unsharpimg

def highboost(img,Unsharp,k):
    high = np.zeros((img.shape[0], img.shape[1]))
    high = img + k*Unsharp
    return high
    
def sobel_bk():
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return sobel_x, sobel_y

def roberts_bk():
    roberts_x = np.array([[0,0,-1],[0,1,0],[0,0,0]])
    roberts_y = np.array([[-1,0,0],[0,1,0],[0,0,0]])
    return roberts_x, roberts_y
    
def binary_bk(img,thre):
    binary_data = np.zeros_like(img)
    height = img.shape[0]
    width = img.shape[1]
    for x in range(height):
        for y in range(width):
            value = img[x][y]
            if value > thre:
                binary_data[x][y]=1
    return binary_data

def BmorphD_bk(img,s):
    p = int((s.shape[0] -1)/2)  #패딩 개수 계산
    height =  img.shape[0]
    width = img.shape[1]
    
    img_p = np.zeros((height+p*2,width+p*2))
    img_p[p:-p,p:-p] = img  #패딩이 적용된 이미지
    img_dil = np.zeros_like(img)
    
    for i in range(img_p.shape[0]-(s.shape[0]-1)):
        for j in range(img_p.shape[1] - (s.shape[0]-1)):
            mul = img_p[i:i+s.shape[0],j:j+s.shape[0]] *s
            sum_mul = sum(sum(mul))
            if sum_mul >0:
                img_dil[i][j] =1
            else:
                img_dil[i][j] =0
    return img_dil

def BmorphE_bk(img,s,count):
    p = int((s.shape[0] -1)/2)  #패딩 개수 계산
    height =  img.shape[0]
    width = img.shape[1]
    
    img_p = np.zeros((height+p*2,width+p*2))
    img_p[p:-p,p:-p] = img  #패딩이 적용된 이미지
    img_er = np.zeros_like(img)
    
    for i in range(img_p.shape[0]-(s.shape[0]-1)):
        for j in range(img_p.shape[1] - (s.shape[0]-1)):
            mul = img_p[i:i+s.shape[0],j:j+s.shape[0]] *s
            sum_mul = sum(sum(mul))
            if sum_mul==count:
                img_er[i][j] =1
            else:
                img_er[i][j] =0
    return img_er

def CmorphD_bk(img,s):
    p = int((s.shape[0] -1)/2)  #패딩 개수 계산
    height =  img.shape[0]
    width = img.shape[1]
    
    img_p = np.zeros((height+p*2,width+p*2))
    img_p[p:-p,p:-p] = img  #패딩이 적용된 이미지
    img_dil = np.zeros_like(img)
    
    for i in range(img_p.shape[0]-(s.shape[0]-1)):
        for j in range(img_p.shape[1] - (s.shape[0]-1)):
            add = img_p[i:i+s.shape[0],j:j+s.shape[0]] + s
            maxV = add.max()
            img_dil[i][j] = maxV
    return img_dil

def CmorphE_bk(img,s):
    p = int((s.shape[0] -1)/2)  #패딩 개수 계산
    height =  img.shape[0]
    width = img.shape[1]
    
    img_p = np.zeros((height+p*2,width+p*2))
    img_p[p:-p,p:-p] = img  #패딩이 적용된 이미지
    img_er = np.zeros_like(img)
    
    for i in range(img_p.shape[0]-(s.shape[0]-1)):
        for j in range(img_p.shape[1] - (s.shape[0]-1)):
            add = img_p[i:i+s.shape[0],j:j+s.shape[0]] - s
            minV = add.min()
            img_er[i][j] = minV
    return img_er

def otsu_bk(img,hist):
    hist_n = hist/(img.shape[0]* img.shape[1])
    w0 = np.zeros(256)
    avg0 = np.zeros(256)
    avg1 = np.zeros(256)
    w0[0] = 0
    avg0[0] = 0
    avg = 0
    vb = np.zeros(256)
    for i in range(0,256):
        avg = avg + i * hist_n[i]
    
    for t in range(1,256):
        w0[t] = w0[t-1] + hist_n[t]
        if w0[t] == 0:
            avg0[t] =0
        else:
            avg0[t] = (w0[t-1]*avg0[t-1] + t*hist_n[t])/w0[t]
        if w0[t] == 1:
            avg1[t] = 0
        else:
            avg1[t] = (avg-w0[t]*avg0[t])/(1-w0[t])
        vb[t] = w0[t]*(1-w0[t])*(avg0[t]-avg1[t])**2
        ind = np.unravel_index(np.argmax(vb, axis=None), vb.shape)
    return ind[0]