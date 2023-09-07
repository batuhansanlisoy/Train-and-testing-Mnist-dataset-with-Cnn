import numpy as np
import os 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Flatten,MaxPooling2D,Activation,Dense,BatchNormalization
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import load_model
import os


#number of classes default 10 because our data consists of ten classes
#input shape default (28,28,1) for tensor


class Mnist:
   
    y_train=[]
    y_test=[]

    def __init__(self,train_path,test_path):
        self.X_train=[]
        self.X_test=[]
        self.y_train=[]
        self.y_test=[]
        self.train_path=train_path
        self.test_path=test_path
        self.num_classes=10
        self.input_shape=(28,28,1)
        
        #begin read data
        self.readData()
    
        
        
    def readData(self):
        self.train=pd.read_csv(self.train_path)
        self.test=pd.read_csv(self.test_path)
    
        self.y_train=self.train["label"]
        self.y_test=self.test["label"]
        
        
        for i in range(len(self.train)):
            
            k=self.train.iloc[i][1:]
            k=np.array(k)
            k=np.reshape(k,(28,28))
            self.X_train.append(k)
        
        
        for j in range(len(self.test)):
            p=self.test.iloc[j][1:]
            p=np.array(p)
            p=np.reshape(p,(28,28))
            self.X_test.append(p)
            
            
        self.preProcces()
            
            
            
    def preProcces(self):
        
        self.X_train=np.array(self.X_train)
        self.X_test=np.array(self.X_test)
        
        self.X_train=self.X_train.reshape(-1,28,28,1)
        self.X_test=self.X_test.reshape(-1,28,28,1)
        
        
        self.y_train=to_categorical(self.y_train,self.num_classes)
        self.y_test=to_categorical(self.y_test,self.num_classes)
                    
        
        self.createModel()
            
            
    def createModel(self):
                
        model = Sequential()
        
        model.add(Conv2D(input_shape = (28,28,1), filters = 64, kernel_size = (3,3)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        
        model.add(Conv2D(filters = 128, kernel_size = (3,3)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        
        
        model.add(Conv2D(filters = 128, kernel_size = (3,3)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        
        
        model.add(Flatten())
        model.add(Dense(units = 256))
        
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(units = self.num_classes))
        model.add(Activation("softmax"))
        
        model.compile(loss = "categorical_crossentropy",optimizer = "rmsprop", metrics = ["accuracy"])

        batch_size=2000
        
        
        
        hist=model.fit(self.X_train,self.y_train,
                       
                      validation_data=(self.X_test,self.y_test),
                
                      epochs=25,
                    
                      batch_size=batch_size)
        
        model.save("mnistModel.h5")
        
        self.visualize(model,hist)
                       
    def visualize(self,model,hist):
               
        y_pred=model.predict(self.X_test)
        y_pred_class=np.argmax(y_pred,axis=1)
        y_true=np.argmax(self.y_test,axis=1)


        cm=confusion_matrix(y_true, y_pred_class)

        sns.heatmap(cm,annot=True,cmap="Greens",fmt=".1f")
        plt.xlabel("predicted")
        plt.ylabel("True")
        plt.title("confusion matrix")
        plt.show()  
        
            
            
        plt.figure()
        plt.plot(hist.history["loss"] ,label="train_loss")
        plt.plot(hist.history["val_loss"],label="val_loss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(hist.history["accuracy"],label="train_accuracy")
        plt.plot(hist.history["val_accuracy"],label="val_accuracy")
        plt.legend()
        plt.show()
            
        
            
cnn=Mnist("C:\\Users\\user\\Desktop\\mnist_proje\mnist_train.csv","C:\\Users\\user\\Desktop\\mnist_proje\mnist_test.csv")







# i create image on photoshop for testing. if you want try it
"""
def testReal():
    model=load_model("C:/Users/user/Desktop/mnist_proje/mnistModel.h5")
    imagesName=os.listdir("C:/Users/user/Desktop/mnist_proje/real_test")
    for i in imagesName:
        img=cv2.imread("C:/Users/user/Desktop/mnist_proje/real_test/"+i,0)
        img=cv2.resize(img,(28,28))
        imgc=img
       
        img=img.reshape(-1,28,28,1)
        x=np.argmax(model.predict(img))
        plt.figure(),plt.title("predicted_value=  "+str(x)),plt.imshow(imgc,cmap="gray")



testReal()
"""


