import matplotlib.pyplot as plt
import matplotlib.image as mpimg # image related ops
import numpy as np
import cv2 # opencv lib
from google.colab.patches import cv2_imshow

img_path="resources/lena.png"
img=mpimg.imread(img_path)  # matplotlib image reads in RGB format
img2 = cv2.imread(img_path) # cv2 reads in BGR format
print(img.shape) # height, width, channels
plt.imshow(img) # show image
plt.axis("off") # hide axis
plt.show()
plt.imshow(img2) # show image
plt.show()
# pour convertir de BGR a RGB
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2_rgb) # show image
# gray scale
gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray') # show image

def visualize_RGB_channels(imgArray=None, figsize=(10,7)):
    # splitting channels
    B,G,R=cv2.split(imgArray)
    # create zero matrix of shape of image
    Z=np.zeros(B.shape,dtype=B.dtype)
    # init subplots
    fig,ax=plt.subplots(2,2,figsize=figsize)
    # plot the actual image and RGB images
    [ax.set_axis_off() for a in ax.ravel()] # hide axis
    ax[0,0].imshow(cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB))
    ax[0,0].set_title("Actual Image")
    ax[0,1].imshow(cv2.merge([R,Z,Z]))
    ax[0,1].set_title("Red Channel")
    ax[1,0].imshow(cv2.merge([Z,G,Z]))
    ax[1,0].set_title("Green Channel")
    ax[1,1].imshow(cv2.merge([Z,Z,B]))
    ax[1,1].set_title("Blue Channel")
    plt.show()

def simple_conv(imgFilter=None,picture=None): # picture and output are gray scale image
    # extract the shape of the img
    rs,cs=picture.shape
    # extract the shape of the filter
    k=imgFilter.shape[0]
    temp=list()
    stride=1
    # resulted image size
    rs_f=(rs-k)//stride +1
    cs_f=(cs-k)//stride +1
    # take vertically down stride across row by row
    for v_striede in range(rs_f):
        for h_stride in range(cs_f):
            targeted_area=imgFilter,picture[v_striede:v_striede+k,h_stride:h_stride+k]
            temp_sum=sum(sum(imgFilter*targeted_area))
            temp.append(temp_sum) # 1-D flat array
    conv_image=np.array(temp).reshape(rs_f,cs_f) # reshape to 2D
    return conv_image

def main():
    visualize_RGB_channels(img)
    random_nums=np.random.randint(0,255,size=(6,6,3)).astype(np.uint8)
    plt.imshow(random_nums)
    plt.title("Random Image")
    plt.show()
    visualize_RGB_channels(random_nums)
    filtered_img=simple_conv(np.array([[1,0,-1],[1,0,-1],[1,0,-1]]),gray_img)
    plt.imshow(filtered_img,cmap='gray')
    plt.title("Filtered Image - Simple Convolution")
    plt.show()
if __name__ == "__main__":
    main()