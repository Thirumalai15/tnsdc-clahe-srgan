import numpy as np
import cv2
import streamlit as st
from skimage.restoration import denoise_wavelet
from skimage.filters import median
from skimage.morphology import disk


def detect_noise(image):
    # Simple noise detection logic
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean = np.mean(image)
    std = np.std(image)

    # Check for Gaussian noise (high std deviation)
    if std > 30:
        return "gaussian"

    # Check for Salt and Pepper noise (many extremes)
    salt_pepper = ((image == 0) | (image == 255)).sum() / image.size
    if salt_pepper > 0.05:
        return "salt_and_pepper"

    # Assume speckle noise if neither condition above is true
    return "speckle"

def remove_noise(image, noise_type):
    if noise_type == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif noise_type == "speckle":
        return denoise_wavelet(image, multichannel=False)
    elif noise_type == "salt_and_pepper":
        return median(image, disk(3))
    else:
        return image

#INTERPOLATION FUNCTION
def interpolate(subBin,LU,RU,LB,RB,subX,subY):
    subImage = np.zeros(subBin.shape)
    num = subX*subY
    for i in range(subX):
        inverseI = subX-i
        for j in range(subY):
            inverseJ = subY-j
            val = subBin[i,j].astype(int)
            subImage[i,j] = np.floor((inverseI*(inverseJ*LU[val] + j*RU[val])+ i*(inverseJ*LB[val] + j*RB[val]))/float(num))
    return subImage

#CLAHE FUNCTION
#ALL UTILITY FUNCTIONS COMBINED INTO ONE FUNCTION
def clahe(img,clipLimit,nrBins=128,nrX=0,nrY=0):
    '''img - Input image
       clipLimit - Normalized clipLimit. Higher value gives more contrast
       nrBins - Number of graylevel bins for histogram
       nrX - Number of contextial regions in X direction
       nrY - Number of Contextial regions in Y direction'''
    h,w = img.shape
    if clipLimit==1:
        return
    nrBins = max(nrBins,128)
    if nrX==0:
        #Taking dimensions of each contextial region to be a square of 32X32
        xsz = 32
        ysz = 32
        nrX = np.ceil(h/xsz).astype(int)#240
        #Excess number of pixels to get an integer value of nrX and nrY
        excX= int(xsz*(nrX-h/xsz))
        nrY = np.ceil(w/ysz).astype(int)#320
        excY= int(ysz*(nrY-w/ysz))
        #Pad that number of pixels to the image
        if excX!=0:
            img = np.append(img,np.zeros((excX,img.shape[1])).astype(int),axis=0)
        if excY!=0:
            img = np.append(img,np.zeros((img.shape[0],excY)).astype(int),axis=1)
    else:
        xsz = round(h/nrX)
        ysz = round(w/nrY)

    nrPixels = xsz*ysz
    xsz2 = round(xsz/2)
    ysz2 = round(ysz/2)
    claheimg = np.zeros(img.shape)

    if clipLimit > 0:
        clipLimit = max(1,clipLimit*xsz*ysz/nrBins)
    else:
        clipLimit = 50

    #makeLUT
    print("...Make the LUT...")
    minVal = 0 #np.min(img)
    maxVal = 255 #np.max(img)

    #maxVal1 = maxVal + np.maximum(np.array([0]),minVal) - minVal
    #minVal1 = np.maximum(np.array([0]),minVal)

    binSz = np.floor(1+(maxVal-minVal)/float(nrBins))
    LUT = np.floor((np.arange(minVal,maxVal+1)-minVal)/float(binSz))

    #BACK TO CLAHE
    bins = LUT[img]
    print(bins.shape)
    #makeHistogram
    print("...Making the Histogram...")
    hist = np.zeros((nrX,nrY,nrBins))
    print(nrX,nrY,hist.shape)
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i*xsz:(i+1)*xsz,j*ysz:(j+1)*ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i,j,bin_[i1,j1]]+=1

    #clipHistogram
    print("...Clipping the Histogram...")
    if clipLimit>0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i,j,nr] - clipLimit
                    if excess>0:
                        nrExcess += excess

                binIncr = nrExcess/nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i,j,nr] > clipLimit:
                        hist[i,j,nr] = clipLimit
                    else:
                        if hist[i,j,nr]>upper:
                            nrExcess += upper - hist[i,j,nr]
                            hist[i,j,nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i,j,nr] += binIncr

                if nrExcess > 0:
                    stepSz = max(1,np.floor(1+nrExcess/nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i,j,nr] += stepSz
                        if nrExcess < 1:
                            break

    #mapHistogram
    print("...Mapping the Histogram...")
    map_ = np.zeros((nrX,nrY,nrBins))
    #print(map_.shape)
    scale = (maxVal - minVal)/float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i,j,nr]
                map_[i,j,nr] = np.floor(min(minVal+sum_*scale,maxVal))

    #BACK TO CLAHE
    #INTERPOLATION
    print("...interpolation...")
    xI = 0
    for i in range(nrX+1):
        if i==0:
            subX = int(xsz/2)
            xU = 0
            xB = 0
        elif i==nrX:
            subX = int(xsz/2)
            xU = nrX-1
            xB = nrX-1
        else:
            subX = xsz
            xU = i-1
            xB = i

        yI = 0
        for j in range(nrY+1):
            if j==0:
                subY = int(ysz/2)
                yL = 0
                yR = 0
            elif j==nrY:
                subY = int(ysz/2)
                yL = nrY-1
                yR = nrY-1
            else:
                subY = ysz
                yL = j-1
                yR = j
            UL = map_[xU,yL,:]
            UR = map_[xU,yR,:]
            BL = map_[xB,yL,:]
            BR = map_[xB,yR,:]
            #print("CLAHE vals...")
            subBin = bins[xI:xI+subX,yI:yI+subY]
            #print("clahe subBin shape: ",subBin.shape)
            subImage = interpolate(subBin,UL,UR,BL,BR,subX,subY)
            claheimg[xI:xI+subX,yI:yI+subY] = subImage
            yI += subY
        xI += subX

    if excX==0 and excY!=0:
        return claheimg[:,:-excY]
    elif excX!=0 and excY==0:
        return claheimg[:-excX,:]
    elif excX!=0 and excY!=0:
        return claheimg[:-excX,:-excY]
    else:
        return claheimg

def clahe_example_UI():
    """
    The UI function to display the CLAHE example
    """
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_type = detect_noise(gray)
        st.write(f"Detected noise type: {noise_type}")

        hist = [0] * 256
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                hist[gray[i, j]] += 1

        cdf = [0] * 256
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + hist[i]

        cdf_norm = [0] * 256
        for i in range(256):
            cdf_norm[i] = cdf[i] / cdf[-1]

        equalized = np.zeros_like(gray)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                equalized[i, j] = 255 * cdf_norm[gray[i, j]]

        clahe_img = clahe(equalized,2,0,0)
        cv2.imwrite('clahe2.png',clahe_img)

        results = st.empty()

        ## Display the results
        with results.container():
            col1, col2 = st.columns(2)

            ## Display the low resolution image
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_column_width=True)

            ## Display the super resolution image
            with col2:
                st.subheader("CLAHE Enhanced image")
                st.image('clahe2.png', use_column_width=True)




# def clahe_example_UI():
#     """
#     The UI function to display the CLAHE example
#     """
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
#         img_array = np.array(image)

#         # Histogram equalization
#         hist = [0] * 256
#         for i in range(img_array.shape[0]):
#             for j in range(img_array.shape[1]):
#                 hist[img_array[i, j]] += 1

#         cdf = [0] * 256
#         cdf[0] = hist[0]
#         for i in range(1, 256):
#             cdf[i] = cdf[i - 1] + hist[i]

#         cdf_norm = [0] * 256
#         for i in range(256):
#             cdf_norm[i] = cdf[i] / cdf[-1]

#         equalized = np.zeros_like(img_array)
#         for i in range(img_array.shape[0]):
#             for j in range(img_array.shape[1]):
#                 equalized[i, j] = 255 * cdf_norm[img_array[i, j]]

#         clahe_img = clahe(equalized, 2, 0, 0)

#         # Normalize the image to the range [0, 1] for display
#         img_array_normalized = img_array / 255.0
#         clahe_img_normalized = clahe_img / 255.0

#         st.image([img_array_normalized, clahe_img_normalized], caption=["Original Image", "CLAHE Enhanced Image"], use_column_width=True)