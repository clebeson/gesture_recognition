import numpy as np
import matplotlib.pyplot as plt 
import pandas as pandas
import pickle as pkl

def pred_uncertainty(prev_label,  label_now, uncert, thr=0.4):
    if uncert > thr:return prev_label
    return label_now
def intersection(s1, s2) : 
  
    # Find the intersection of the two sets  
    intersect = s1 & s2 ; 
  
    return intersect;  
  
  
# Function to return the Jaccard index of two sets  
def jaccard_index(s1, s2) : 
      
    # Sizes of both the sets  
    size_s1 = len(s1);  
    size_s2 = len(s2);  
  
    # Get the intersection set  
    intersect = intersection(s1, s2);  
  
    # Size of the intersection set  
    size_in = len(intersect);  
  
    # Calculate the Jaccard index  
    # using the formula  
    jaccard_in = size_in  / (size_s1 + size_s2 - size_in);  
  
    # Return the Jaccard index  
    return jaccard_in;  
def calc_uncertainties(probs):
    #vr= np.argmax(probs, axis= 2).mean(1) #variation ratio
    if len(probs.shape) > 2:
        mean = probs.mean(1)
        h = -(mean*np.log(mean)).sum(1) #entropy
        mi = -(probs*np.log(probs)).sum(2).mean(1)#mutual information
    else: 
        mean = probs.mean(0)
        h = -(mean*np.log(mean)).sum(0) #entropy
        mi = -(probs*np.log(probs)).sum(1).mean(0)#mutual information
    
    return h,mi,h+mi

data = pkl.load(open("prediction_spotting3_mc.pkl","rb"))
h,_,_ = calc_uncertainties(data[0]["probs"])
predict = data[0]["pred"]
prev = 1
pred_unc = []
for unc,pred in zip(h,predict):
    now = pred_uncertainty(prev,pred,unc)
    pred_unc.append(now)
    prev = now

pred_unc = np.array(pred_unc)
limit = -1
print((data[0]["label"]==pred_unc).sum()/float(len(pred_unc)))
#plt.plot(h[:limit]/h[:limit].max())
plt.plot(pred_unc[:limit])
# plt.plot(data[0]["probs"].mean(1).max(1)[:limit])
plt.plot(data[0]["label"][:limit])
plt.legend([ "uncert","pred", "label" ])
label = data[0]["label"]
print("JI:",sum(label & pred_unc)/sum(label | pred_unc))
plt.show()















