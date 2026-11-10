# Developing a Neural Network Classification Model
# NAME:DEEPAK SUDHARSHAN B      
# REG NO:212225230045
## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import all the required libraries such as PyTorch, Pandas, NumPy, Matplotlib, Scikit-learn, andSeaborn.

### STEP 2: 

Load the customer segmentation dataset from the CSV file and remove unnecessary columnssuch as the customer ID.

### STEP 3: 

Handle missing values in the dataset by replacing them with suitable values. Fill missing workexperience values with 0 and family size values with the median.

### STEP 4: 

Convert all categorical features into numerical values using Label Encoding so that they can beprocessed by the neural network.

### STEP 5: 

Encode the target column (Segmentation) into numerical classes and separate the dataset intoinput features (X) and target labels (Y).

### STEP 6: 

Split the dataset into training and testing sets. Apply feature scaling using StandardScaler tonormalize the input data.

### STEP 7: 

Convert the training and testing data into PyTorch tensors and create TensorDatasets andDataLoaders for batch processing.

### STEP 8: 

Define a feedforward neural network with four fully connected layers and ReLU activationfunctions.

### STEP 9: 

Initialize the model, define the Cross Entropy Loss function, and configure the Adam optimizer.

### STEP 10: 

Train the neural network for multiple epochs by performing forward propagation, losscalculation, backpropagation, and weight updates.

### STEP 11: 

Use the trained model to predict the segmentation classes for the test dataset.

### STEP 12: 

Evaluate the model using Accuracy Score, Confusion Matrix, and Classification Report.

### STEP 13: 

Visualize the confusion matrix using a Seaborn heatmap to analyze the classificationperformance of the model.


## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5))  
])

train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,
transform=transform)
test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,
transform=transform)

im,lbl=train_set[0]
print(im.shape)
print(len(train_set))
print(len(test_set))

trl=DataLoader(train_set,batch_size=32,shuffle=True)
tstl=DataLoader(test_set,batch_size=32,shuffle=False)

class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x

from torchsummary import summary

model=CNNclassifier1()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model.to(device)


summary(model,input_size=(1,28,28))

epochs=3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))

t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

with torch.no_grad():
    img1,label=test_set[0]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")
```


### OUTPUT

<img width="267" height="82" alt="image" src="https://github.com/user-attachments/assets/7d750c0f-1b44-47ff-a44c-ef8857f312db" />

<img width="692" height="537" alt="image" src="https://github.com/user-attachments/assets/2c1aef9b-8e1a-41bd-b72b-d3f5d77c9b4b" />

<img width="342" height="112" alt="image" src="https://github.com/user-attachments/assets/97414f88-8632-4258-8f4c-37aeefc1c24b" />

## Confusion matrix:

<img width="822" height="687" alt="image" src="https://github.com/user-attachments/assets/898151de-a9dc-4a43-a4f2-835283fcd299" />

## Classification report:

<img width="798" height="412" alt="image" src="https://github.com/user-attachments/assets/05fbcd58-2eb9-4a3e-a84d-d891efd7d8df" />



### New Sample Data Prediction

<img width="630" height="676" alt="image" src="https://github.com/user-attachments/assets/d989a9c8-3147-4ef5-95e9-a729af315e1e" />


## RESULT
The Convolutional Neural Network was successfully developed and trained for image classification using PyTorch.

