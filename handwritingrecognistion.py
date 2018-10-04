from scipy import misc  # to import image
from sklearn import datasets
from sklearn.svm import SVC  # super vector machine and super vector class


digits=datasets.load_digits()
#print(digits) #to check the data format
features=digits.data    #seperating data and labels
labels=digits.target
#print(features,labels)


clf = SVC(gamma=0.001)
clf.fit(features,labels)
#print(features.shape) #to check the dimention of features
#print(clf.predict([features[-1]])) #check corresponding features and labels value

img = misc.imread("Python programs/test_pic/test_pic_9.jpg")

img = misc.imresize(img,(8,8))  #resizing the img to 8*8 pixels from 64p
img = img.astype(digits.images.dtype)  #converting dtypr to float64 from uint32
img = misc.bytescale(img,high=16,low=0)  #converting ing size to 16byte from 255bytes
#print(img)  #converted pixels img

x_test = []

for eachRow in img:
    for eachPixel in eachRow:
        x_test.append(sum(eachPixel)/3.0)  #making 1d 64bit array
        
print(clf.predict([x_test]))   #predicting the image digit value
        
        