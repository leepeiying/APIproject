from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
#tkinter
window = tk.Tk()
window.title('AIP 60947089S')
window.geometry('1150x650')
#打開影像
def Open_Img():
    global filename,img,nimg,img_png,label_Img,label_Img2
    filename = tkinter.filedialog.askopenfilename(filetypes=[("Image files",("*.jpg","*.jpeg","*.png","*.ppm","*.bmp"))])
    img = Image.open(filename)
    img.save('Original Image.jpg')
    main()
#顯示影像於tkinter    
def lable_Img(img1,img2):
    global Imgh1,label_Img,Imgh2,imgtk1,imgtk2
    print('INto lable_Img')
    Imgh1 = Image.open(img1)
    print('Imgh1',Imgh1)
    Imgh1 = Imgh1.resize((500,500), Image.ANTIALIAS)
    Imgh2 = Image.open(img2)
    Imgh2 = Imgh2.resize((500,500), Image.ANTIALIAS)
    imgtk1 = ImageTk.PhotoImage(Imgh1)
    imgtk2 = ImageTk.PhotoImage(Imgh2)
    label_Img = tk.Label(window, image=imgtk1)
    label_Img.place(x=20,y=100) # 第一張圖的位置
    plt.cla()#清除之前的圖片
    label_Img= tk.Label(window, image=imgtk2)
    label_Img.place(x=600,y=100) #第二張圖的位置
    plt.cla()

#影像灰度化
def convert_to_gray(img):
    global rows,cols,layers,gray
    if len(img.shape)==3:
        rows,cols,layers=img.shape
        gray=img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114
    else:
        rows,cols=img.shape
        gray=img[:,:]
    return gray

#直方圖均化algo 處理亮度不均   
def make_histogram(img):
    H = np.zeros(256, dtype=int)
    for g in range(img.size):
        H[img[g]] += 1
    return H

def make_cumsum(H):
    HC = np.zeros(256, dtype=int)
    HC[0] = H[0]
    for g in range(1, H.size):
        HC[g] = HC[g-1] + H[g]
    return HC

def make_mapping(HC,H):
    T = np.zeros(256, dtype=int)
    G = 256
    gmin = min(H)
    for i in range(len(H)):
        if H[i]>0:
            gmin = i
            break
    Hmin = HC[gmin]
    for g in range(256):
        T[g] = round(((HC[g]-Hmin)/(HC[255]-Hmin))*(G-1))
    return T
#step5
def apply_mapping(img, T):
    new_image = np.zeros(img.size, dtype=int)
    for g in range(img.size):
        new_image[g] = T[img[g]]
    return new_image
def histogram_equalization():
    global pillow_img,img,IMG_W,IMG_H,Img_gray

    #pillow_img = Image.open('grayimage.jpg')
    pillow_img = Img_gray.convert("L")#讀取灰階影像轉灰階

    IMG_W, IMG_H = pillow_img.size#要用上方指令才能讀出此

    img = np.array(pillow_img).flatten()#變一維
    histogram = make_histogram(img)
    cumsum = make_cumsum(histogram)
    #mapping = make_mapping(cumsum)
    mapping = make_mapping(cumsum,histogram)
    new_image = apply_mapping(img, mapping)
    
    output_image = Image.fromarray(np.uint8(new_image.reshape((IMG_H, IMG_W))))
    output_image.save('equalization.jpg')#直方圖均化的圖片
    plt.set_cmap('gray')
    plt.title('equalization')
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()
#機器學習
def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii
#臉部辨識
def face_detection(img):
    global img_copy,face_cascade,faces,pathf
    img_copy = img.copy()#灰階均化
    print('img_copy:',img_copy)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    print('face_cascade:',face_cascade)
    print('eye_cascade:',eye_cascade)
    faces = face_cascade.detectMultiScale(img,1.1,4)
    print('faces:',faces)
    '''
    for(x,y,w,h)in faces:
        img_copy = cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,255),2)
    return img_copy
    '''
    for(x,y,w,h) in faces:
        roi_gray = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.02,3)
        img_copy = cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,0),2)
        for(ex,ey,ew,eh) in eyes:
            img_copy = cv2.rectangle(img_copy,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
    return img_copy
#主程式
def main():
    global img_data,gray,Img_gray,img_face,output_image,img1,img2
    img_data = np.array(img)#原圖影像
    gray = convert_to_gray(img_data)#灰階數值
    print('grayimage:',gray)
    Img_gray = Image.fromarray(gray)#數值轉影像
    print('Img_gray:',Img_gray)
    plt.imsave('grayimage.jpg',Img_gray,cmap=cm.gray)
    plt.title('Convert to Gray')
    plt.imshow(Img_gray)
    plt.axis("off")
    plt.show()
    histogram_equalization()#灰階影像去做均化
    img1 = cv2.imread('equalization.jpg')
    integral_img = integral_image(img1)
    print('integral_image:',integral_img)
    img2 = face_detection(img1)
    plt.imsave('face_detection.jpg',img2)
    plt.title('face_detection')
    plt.imshow(img2)
    plt.axis("off")
    plt.show()
    lable_Img('Original Image.jpg','face_detection.jpg')
    os.remove('Original Image.jpg')
    os.remove('grayimage.jpg')
    os.remove('equalization.jpg')
    os.remove('face_detection.jpg')
    
#按鈕
btn_Open = tk.Button(window,text='選擇圖像',width=10,height=1,command=Open_Img)
btn_Open.place(x=100, y=20)
#持續視窗   
window.mainloop()