import csv
import numpy as np
import numpy.matlib as mp
import matplotlib.pyplot as plt
def accuracy(X,theta1,mu1,sd1,y,m,now):
          dev=np.zeros((259,now))
          y1=np.zeros((259,1))
          for i in range(259):
                    for j in range(now):
                              dev[i][j]=(X[i][j]-mu1[j])/(sd1[j])
         
          y1=sigmoid(np.dot(dev,theta1))
          posclass=0
          negclass=0
          for i in range(m):
                    if(y1[i]>0.5 and y[i]==1):
                              posclass=posclass+1
                    elif(y1[i]<0.5 and y[i]==0):
                              negclass=negclass+1
                    else:
                              posclass=posclass
                              negclass=negclass
                    
                    
          acc= (posclass+negclass)/m
          return acc
def tst_accuracy(X,theta1,mu1,sd1,y,m,now):
          dev=np.zeros((50,now))
          y1=np.zeros((50,1))
          for i in range(50):
                    for j in range(now):
                              dev[i][j]=(X[i][j]-mu1[j])/(sd1[j])
          #print(dev)
          y1=sigmoid(np.dot(dev,theta1))
          #print(y1)
          posclass=0
          negclass=0
          for i in range(m):
                    if(y1[i]>0.5 and y[i]==1):
                              posclass=posclass+1
                    elif(y1[i]<0.5 and y[i]==0):
                              negclass=negclass+1
                    else:
                              posclass=posclass
                              negclass=negclass
                    
                    
          acc= (posclass+negclass)/m
          return acc
def normalize(X):
          X_norm = X
          [r,c]=np.shape(X)
          mu = np.zeros([c,1])
          sigma = np.zeros([c,1])    
          for i in range(c):
                    mu[i] = np.mean(X[:,i])
                    sigma[i] = np.std(X[:,i])
                    X_norm[:,i] =(X[:,i]-mu[i])/sigma[i];
          return [X_norm,mu,sigma]
def sigmoid(z):
          return (1/(1+np.exp(-z)))
a=np.zeros((259,1))
b=np.zeros((259,1))
c=np.zeros((259,1))
d=np.zeros((259,1))
e=np.zeros((259,1))
f=np.zeros((259,1))
g=np.zeros((259,1))
h=np.zeros((259,1))
p=np.zeros((259,1))
j=np.zeros((259,1))
k=np.zeros((259,1))
l=np.zeros((259,1))
m=np.zeros((259,1))
n=np.zeros((259,1))
o=np.zeros((259,1))
y=np.zeros((259,1))
X1=np.zeros((259,15))
i=0
#####Read Training data#######
with open("lung_cancer_train.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for lines in csv_reader:
      b[i]=lines['age']
      if(lines['gender']=='M'):
                a[i]=1
      else:
                a[i]=0
      if(lines['lung_cancer']=='true'):
                y[i]=1
      else:
                y[i]=0
      c[i]=lines['smoking']
      d[i]=lines['yellow_fingers']
      e[i]=lines['anxiety']
      f[i]=lines['peer_pressure']
      g[i]=lines['chronic_disease']
      h[i]=lines['fatigue']
      p[i]=lines['allergy']
      j[i]=lines['wheezing']
      k[i]=lines['alcohol_consuming']
      l[i]=lines['coughing']
      m[i]=lines['shortness_of_breath']
      n[i]=lines['swallowing_difficulty']
      o[i]=lines['chest_pain']
      i=i+1

for i in range(259):
  X1[i][0]=a[i]
  X1[i][1]=b[i]
  X1[i][2]=c[i]
  X1[i][3]=d[i]
  X1[i][4]=e[i]
  X1[i][5]=f[i]
  X1[i][6]=g[i]
  X1[i][7]=h[i]
  X1[i][8]=p[i]
  X1[i][9]=j[i]
  X1[i][10]=k[i]
  X1[i][11]=l[i]
  X1[i][12]=m[i]
  X1[i][13]=n[i]
  X1[i][14]=o[i]
now=16
X3=np.ones((259,now))
for i in range(259):
          for j in range(now-1):
                    X3[i][j+1]=X1[i][j]
[X2,mu,sigma]=normalize(X1)
X=np.ones((259,now))
for i in range(np.size(X,0)):
          X[i][1]=X2[i][0]
          X[i][2]=X2[i][1]
          X[i][3]=X2[i][2]
          X[i][4]=X2[i][3]
          X[i][5]=X2[i][4]
          X[i][6]=X2[i][5]
          X[i][7]=X2[i][6]
          X[i][8]=X2[i][7]
          X[i][9]=X2[i][8]
          X[i][10]=X2[i][9]
          X[i][11]=X2[i][10]
          X[i][12]=X2[i][11]
          X[i][13]=X2[i][12]
          X[i][14]=X2[i][13]
          X[i][15]=X2[i][14]
#init_theta= np.zeros((now,1))
init_theta=np.random.randn(now,1)*0.01
theta= np.zeros((now,1))
m= len(y) #traininig examples
J=0
alpha=0.1
grad=np.zeros((now,1))
cost1=[]
for i in range(2000):
          theta=init_theta
          #print(np.shape(theta))
          h= np.dot(X,theta)
          g= np.log(sigmoid(h))
          g1= np.log(1-sigmoid(h))
          temp1= -(np.dot(np.transpose(y),g))
          temp2= -(np.dot((1-np.transpose(y)),g1))
          J=(temp1+temp2)/m
          #print(J)
          cost1.append(J)
          h = sigmoid(X.dot(theta.reshape(-1,1)))
          grad = (1/m)*X.T.dot(h-y) 
          grad.flatten()
          theta=  init_theta -(alpha*np.transpose(grad))
          init_theta= np.transpose(theta[0])
#print(theta)
#print(J)
plt.plot(range(len(cost1)),cost1)
plt.xlabel('No. of iterations')
plt.ylabel('Cost function')
plt.title('Logistic regression without optimization')
plt.show()
xt= [1,0,59,1,1,2,1,2,1,2,1,2,1,2,1,1]
theta1= np.zeros((now,1))
for i in range(now):
          theta1[i][0]= theta[0][i]
print(theta1)
mu1=np.zeros((now,1))
sd1=np.zeros((now,1))
mu1[0]=0
sd1[0]=1
for i in range(now-1):
          mu1[i+1]=mu[i]
          sd1[i+1]=sigma[i]
#print(mu1)
#print(sd1)
dev=np.zeros((now,1))
for i in range(now):
          dev[i]=(xt[i]-mu1[i])/(sd1[i])
#print(dev)
pred= sigmoid(np.dot(np.transpose(dev),theta1))
print('Training Accuracy')
print(accuracy(X3,theta1,mu1,sd1,y,m,now))  
#######test data ###########
at=np.zeros((50,1))
bt=np.zeros((50,1))
ct=np.zeros((50,1))
dt=np.zeros((50,1))
et=np.zeros((50,1))
ft=np.zeros((50,1))
gt=np.zeros((50,1))
ht=np.zeros((50,1))
pt=np.zeros((50,1))
jt=np.zeros((50,1))
kt=np.zeros((50,1))
lt=np.zeros((50,1))
mt=np.zeros((50,1))
nt=np.zeros((50,1))
ot=np.zeros((50,1))
yt=np.zeros((50,1))
X1t=np.zeros((50,15))
i=0
#####Read Test data#######
with open("lung_cancer_test.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for lines in csv_reader:
      bt[i]=lines['age']
      if(lines['gender']=='M'):
                at[i]=1
      else:
                at[i]=0
      if(lines['lung_cancer']=='true'):
                yt[i]=1
      else:
                yt[i]=0
      ct[i]=lines['smoking']
      dt[i]=lines['yellow_fingers']
      et[i]=lines['anxiety']
      ft[i]=lines['peer_pressure']
      gt[i]=lines['chronic_disease']
      ht[i]=lines['fatigue']
      pt[i]=lines['allergy']
      jt[i]=lines['wheezing']
      kt[i]=lines['alcohol_consuming']
      lt[i]=lines['coughing']
      mt[i]=lines['shortness_of_breath']
      nt[i]=lines['swallowing_difficulty']
      ot[i]=lines['chest_pain']
      i=i+1

for i in range(50):
  X1t[i][0]=at[i]
  X1t[i][1]=bt[i]
  X1t[i][2]=ct[i]
  X1t[i][3]=dt[i]
  X1t[i][4]=et[i]
  X1t[i][5]=ft[i]
  X1t[i][6]=gt[i]
  X1t[i][7]=ht[i]
  X1t[i][8]=pt[i]
  X1t[i][9]=jt[i]
  X1t[i][10]=kt[i]
  X1t[i][11]=lt[i]
  X1t[i][12]=mt[i]
  X1t[i][13]=nt[i]
  X1t[i][14]=ot[i]
now=16
X3t=np.ones((50,now))
for i in range(50):
          for j in range(now-1):
                    X3t[i][j+1]=X1t[i][j]
m= len(yt) #testing examples
print('Testing accuracy')
print(tst_accuracy(X3t,theta1,mu1,sd1,yt,m,now))  
