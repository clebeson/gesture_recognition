import matplotlib.pyplot as plt
import pickle
import numpy as np
import pims
import cv2
from glob import glob
from itertools import product
import matplotlib as mpl 
from sklearn.metrics import confusion_matrix, accuracy_score
mpl.style.use("seaborn")


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


def generate_confusion_matrix( label,pred, class_names):
        
        def plot_confusion_matrix(cm, classes,
                                    normalize=True,
                                    title='Confusion matrix (%)',
                                    cmap=plt.cm.Blues):
                """
                This function prints and plots the confusion matrix.
                Normalization can be applied by setting `normalize=True`.
                """
                if normalize:
                    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    print("Normalized confusion matrix")
                else:
                    print('Confusion matrix, without normalization')

                print(np.diag(cm))

                plt.imshow(cm, interpolation='nearest', cmap=cmap)
                plt.title(title)
                plt.colorbar()
                plt.grid(False)
                
                tick_marks = np.arange(len(classes))
               
          
                plt.xticks(tick_marks, classes, rotation=90)
                plt.yticks(tick_marks, classes)

                # fmt = '.1f' if normalize else 'd'
                
                thresh = cm.max() / 2.
                symbol = "%" if normalize else ""
                for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                    
                    if cm[i, j] > 0:
                        print(i,j)
                        #if i == j:
                        plt.text(j, i, format(cm[i, j], ".1f" if cm[i, j] > int(cm[i, j]) else ".0f"),
                                    horizontalalignment="center", verticalalignment="center", fontsize=11,
                                    color="white" if cm[i, j] > thresh else "black")

                # plt.tight_layout()
                # plt.ylabel('Real')
                # plt.xlabel('Predicted')
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(label,pred)
        np.set_printoptions(precision=2)
        

        # # Plot normalized confusion matrix
       
        # plt.figure(figsize=(20,10))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            # title='Normalized confusion matrix'
                            )
    
        # plt.savefig("test.svg", format="svg")
        #plt.savefig("./confusion_matrix.png") #Save the confision matrix as a .png figure.
        plt.show()



def plot_only_weights(weights, labels, preds):
    fig, axs = plt.subplots(5, 3)
    axs = axs.reshape(-1)
    # print(axs.shape)

    for ax, weight,l, p in zip(axs,weights,labels, preds):
        # print(ax)
        weight = np.array([w for w in weight if w is not None])
        print(weight.shape)
        # label = np.array([l for l,w in zip(label,weight) if w is not None])
        ind = np.arange(len(weight)) # the x locations for the groups
        mov, hand = weight[:,0], weight[:,1]
        width = 1

        ax.bar(ind, mov, width)
        ax.bar(ind, hand, width,bottom=mov)

        ax.set_title('Label {}, Predito {}'.format(l, p), fontsize=12, weight = 'bold')
        # plt.yticks(np.arange(11)/10)
        # plt.plot(label>0, color="orange")
        # plt.plot(pred>0, color="k")
    plt.legend(labels=['Peso para o Movimento', 'Peso para o Contexto (mãos)'], frameon=True, fancybox=True, loc = "upper right", borderpad=1,shadow=True)
    fig.suptitle("Pesos do soft-attention por amostra (BStar iRGB$_{hand}$)", fontsize=20)
    # fig.suptitle("Title centered above all subplots", fontsize=14)
    fig.text(0.5, 0.04, 'Amostras' , fontsize=20, weight = 'bold', ha='center')
    fig.text(0.04, 0.5, 'Pesos', fontsize=20, weight = 'bold', va='center', rotation='vertical')


def plot_weights(weights, label,pred, type = "Spotting"):
    # for ax, weight in zip(axs,weights):
    weights_filter = []
    if type == "Spotting":
        w = np.array([w[:,0] for w,l in zip(weights,label) if l>0])

    else:
        w = np.array([w[:,0]*l for w,l in zip(weights, np.where(label>0,1,0))])
        
    mov, hand = w[:,0], w[:,1]
    label = np.array([l for l,w in zip(label,w) if w is not None])

    ind = np.arange(len(w)) # the x locations for the groups
    width = 1

    plt.bar(ind, mov, width)
    plt.bar(ind, hand, width,bottom=mov)
    plt.ylabel('Pesos' , fontsize=16, weight = 'bold')
    plt.xlabel('Amostras', fontsize=16, weight = 'bold')
    plt.title('Modelo de {}'.format(type), fontsize=16, weight = 'bold')
    # plt.yticks(np.arange(11)/10)
    # plt.plot(label>0, color="orange")
    # plt.plot(pred>0, color="k")
    plt.legend(labels=['Peso para o Movimento', 'Peso para o Contexto (mãos)'], frameon=True, fancybox=True, loc = "upper right", borderpad=1,shadow=True)

def test():
    import numpy as np
    import scipy.misc as m
    import pims
    import generate_iteractive_starRGB as star
    star = star.DynamicStarRGB(max_size = None)
    stars,hands, labels = star.get_images("Sample0004_163_200_13.mp4")
    import matplotlib.pyplot as plt
    for i in range(len(stars)):
        plt.imshow(stars[i])
        plt.show()
    # print(stars.dtype)
def plot_hands():
    hand_files = glob("samples/*.npy")
    files = [1,3,15]
    print(hand_files[0].replace(".npy","").split("_")[-1])
    hand_files = [file for file in hand_files if (not file.endswith("_hands.npy") and int(file.replace(".npy","").split("_")[-1]) in files)]
    print(len(hand_files))
    grid = (3,10)
    size = grid[0]*grid[1]
    for hand_file in hand_files:
        print(hand_file)
        hands = np.uint8(np.load(hand_file))
        print(len(hands))
        for i in range(len(hands)//size):
            mosaico = get_hand_mosaico(hands[i*size:i*size+size],grid)
            plt.imshow(mosaico)
            plt.grid(False)
            plt.show()                   

def get_hand_mosaico(hands, grid = (10,10)):
    space = 10
    image_h, image_w = hands[0].shape[0] , hands[0].shape[1] 
    h,w = image_h + space, image_w + space
    mosaico = np.zeros((grid[0]*h,grid[1]*w,3),dtype = np.uint8)
    idx_hand = 0
    for i in range(grid[0]):
        for j in range(grid[1]):
            if idx_hand < len(hands):
                x = i if i == 0 else i * h
                y = j if j == 0 else j * w 
                hand = hands[idx_hand]
                images = [hand[:,:image_w//2], hand[:,image_w//2:]]
                hand_left = images[0]
                hand_right = images[1]
                images[0] = hand_right[:, ::-1,:]
                images[1] = hand_left[:, ::-1,:]
                hand = np.concatenate(images,1)
                hand[:,image_w//2-1:image_w//2+1] = 255
                mosaico[x:x+image_h,y:y+image_w] = hand[:,::-1]
                idx_hand += 1
    return mosaico
    

# Function to return the Jaccard index of two sets  
def jaccard_index() :
    with open("results/results_anticipation_RTM_8691.pkl","rb") as file:   
        predictions = pickle.load(file)

    JI_i = 0
    JI_u = 0
    for prediction in predictions:
        pred =  np.array(prediction["pred"])
        label = np.array(prediction["label"])
        seqlen = len(pred)
        pred = split_itervals(pred)
        label = split_itervals(label)
        JI_i += gesture_overlap(pred,label,seqlen)
        JI_u += 1
    print(JI_i/JI_u)


    
def split_itervals(pred):
    predictions = []
    for i in range(1,len(pred)):
        if pred[i] > 0 and pred[i-1] == 0:
            b = i
        elif pred[i] == 0 and pred[i-1] > 0 and i-b > 5:           
            predictions.append([pred[i-1],b,i+50,])

    return predictions

 


def gesture_overlap(pred, labels, seqlenght):
    """ Evaluate this sample agains the ground truth file """
    maxGestures=20

    # Get the list of gestures from the ground truth and frame activation
    gtGestures = []
    binvec_gt = np.zeros((maxGestures, seqlenght))
    for row in labels:
        binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
        gtGestures.append(int(row[0]))

    # Get the list of gestures from prediction and frame activation
    predGestures = []
    binvec_pred = np.zeros((maxGestures, seqlenght))
    for row in pred:
        binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
        predGestures.append(int(row[0]))

    # Get the list of gestures without repetitions for ground truth and predicton
    gtGestures = np.unique(gtGestures)
    predGestures = np.unique(predGestures)

    # Find false positives
    falsePos=np.setdiff1d(gtGestures, np.union1d(gtGestures,predGestures))

    # Get overlaps for each gesture
    overlaps = []
    for idx in gtGestures:
        intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
        aux = binvec_gt[idx-1] + binvec_pred[idx-1]
        union = sum(aux > 0)
        overlaps.append(intersec/union)

    # Use real gestures and false positive gestures to calculate the final score
    return sum(overlaps)/(len(overlaps)+len(falsePos))

        



def split_predicition(pred, shift = 0):
    predictions = np.zeros(len(pred))
    # predictions = []
    for i in range(1,len(pred)):
        if pred[i] > 0 and pred[i-1] == 0:
            b = i
        elif pred[i] == 0 and pred[i-1] > 0:           
            predictions[b-shift:i] = pred[i-1] #if i-b > 5 else 0
            # predictions.append(pred[i-1])
    # predcc = []
    # for i in range(len(predictions)):
    #     if i== 0 or (predictions[i] != predcc[-1]):
    #         predcc.append(predictions[i])


            


    return np.array(predictions)

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def calc_anticipation_acc():
    class_names = ["no gesture","vattene","vieniqui","perfetto","furbo","cheduepalle","chevuoi","daccordo","seipazzo","combinato","freganiente","ok","cosatifarei","basta","prendere","noncenepiu","fame","tantotempo","buonissimo","messidaccordo","sonostufo","not anticipated"]
    with open("results/results_anticipation_RTM_8691.pkl","rb") as file:   
        predictions = pickle.load(file)
  

    org = (20, 30) 
    org_box = (30, 30) 
    fontScale = 1
    color_bue = (200, 0, 0)
    color_green = (0,200, 0)
    color_red = ( 0, 0, 200)

    thickness = 2 
    corrects = 0
    total = 0
    JI_i,JI_u = 0,0
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('results/completo.avi', fourcc, 40.0, (640,480))
    corrects_ant =  0
    total_ant = 0.0
    print(len(predictions))
    for prediction in predictions:
        pred =  np.array(prediction["pred"])
        label = np.array(prediction["label"])
        JI_i += sum(pred==label)
        JI_u += len(label)
        probs = prediction["probs"]

        file = prediction["file"]
        acc = sum(pred==label)/len(pred)
        sample = file.split("/")[-1]
        corrects += sum(pred==label)
        acc = sum(pred==label)/len(label)
        total += len(pred)             
        print("{} -> acc {:.2f}%".format(sample,acc*100))
        # video = pims.Video("samples/{}_color.mp4".format(sample))
        label_before = 0
        count = 0
        anticipated_frame = False
        anticipated_gesture = False
        showed_ant = False
        for p, l, prob in zip(pred,label,probs):
            _,_,mi = calc_uncertainties(prob)
            if label_before == 0 and l>0:
                label_before = 1
                count = 1
                anticipated_frame= False
                anticipated_gesture = False
                showed_ant = False
            elif label_before == 1 and l==0:
                label_before = 0
                count = 0
                total_ant += 1
                anticipated_gesture = False
                anticipated_frame = False
                showed_ant = False
            elif l >0:
                    count += 1
                    if (p>0) and (mi<=0.1) and (not anticipated_frame):
                        anticipated_frame = True
                        anticipated_gesture = p
                        
            if anticipated_frame and not showed_ant:
                corrects_ant += 1 if p==l else 0
                showed_ant = True

                        
    print("{} {} {:.2f}\%".format(total_ant, corrects_ant, (corrects_ant/total_ant) *100))




def calc_recognition_acc():
    with open("results/results_anticipation_RTM_8691.pkl","rb") as file:   
        predictions = pickle.load(file)
  
    corrects = 0
    total = 0
    JI_i = 0
    JI_u = 0
    for prediction in predictions:
        pred =  np.array(prediction["pred"])
        label = np.array(prediction["label"])
        label_before = 0
        pred_before = 0
        pred = split_predicition(pred,  shift = 0)
        # label = split_predicition(label,  shift = 0)
        JI_i += sum(np.logical_and(label>0,pred==label))
        JI_u +=  sum(np.logical_or(label>0,pred>0))

        for p, l in zip(pred,label):
            if pred_before >0 and p==0:
                corrects += 1 if pred_before==label_before else 0
                total += 1
            label_before = l
            pred_before = p
    print(JI_i/JI_u)





                        
    print("{} {} {:.2f}\%".format(total, corrects, (corrects/total) *100))


def anticipation_video():
    class_names = ["no gesture","vattene","vieniqui","perfetto","furbo","cheduepalle","chevuoi","daccordo","seipazzo","combinato","freganiente","ok","cosatifarei","basta","prendere","noncenepiu","fame","tantotempo","buonissimo","messidaccordo","sonostufo","not anticipated"]
    best_samples = ["Sample0913","Sample0742","Sample0707","Sample0777"] 
    with open("results/results_anticipation_RTM_8691.pkl","rb") as file:   
        predictions = pickle.load(file)
  

    org = (20, 30) 
    org_box = (30, 30) 
    fontScale = 1
    color_bue = (200, 0, 0)
    color_green = (0,200, 0)
    color_red = ( 0, 0, 200)

    thickness = 2 
    corrects = 0
    total = 0
    JI_i,JI_u = 0,0
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('results/completo.avi', fourcc, 40.0, (640,480))
    for s in best_samples:
        for prediction in predictions:
            pred =  np.array(prediction["pred"])
            label = np.array(prediction["label"])
            JI_i += sum(pred==label)
            JI_u += len(label)
            probs = prediction["probs"]
            weights_clf = prediction["weights_clf"]
            weights_spt = prediction["weights_spt"]
            file = prediction["file"]
            acc = sum(pred==label)/len(pred)
            sample = file.split("/")[-1]
            corrects += sum(pred==label)
            acc = sum(pred==label)/len(label)
            total += len(pred)
            if sample == s:                
                print("{} -> acc {:.2f}%".format(sample,acc*100))
                video = pims.Video("samples/{}_color.mp4".format(sample))
                label_before = 0
                count = 0
                anticipated_frame = False
                anticipated_gesture = False
                showed_ant = False
                for frame, p, l, prob in zip(video,pred,label,probs):
                    _,_,mi = calc_uncertainties(prob)
                    if label_before == 0 and l>0:
                        label_before = 1
                        count = 1
                        anticipated_frame= False
                        anticipated_gesture = False
                        showed_ant = False
                    elif label_before == 1 and l==0:
                        label_before = 0
                        count = 0
                        if not showed_ant:

                                anticipated_frame = 1
                                anticipated_gesture = 21
                                color =  color_red
                        else:
                            anticipated_gesture = False
                            anticipated_frame = False
                        showed_ant = False
                    elif l >0:
                            count += 1
                            if (p>0) and (mi<=0.2) and (not anticipated_frame):
                                showed_ant = False
                                anticipated_frame = count
                                anticipated_gesture = p
                                color = color_green if p==l else color_red
                    elif l == 0 and anticipated_gesture == 21:
                        label_before = 0
                        count = 0
                        anticipated_gesture = False
                        anticipated_frame = False
                        showed_ant = False
                            

                    

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.rectangle(frame, (0,0), (640,70), (200,200,200), cv2.FILLED)
                    frame = cv2.putText(frame, "Label:", (10,30), font, fontScale, (0,0,0), thickness) 
                    frame = cv2.putText(frame, "video 2x", (530,20), font, 0.8, (0,0,0)) 
                    frame = cv2.putText(frame, "Antic.:", (10,60), font, fontScale, (0,0,0), thickness) 
                    frame = cv2.putText(frame, "u=0.2", (530,50), font, 0.8, (0,0,0))
                    frame = cv2.putText(frame, "{}".format(class_names[l]), (120,30), font, fontScale, (0,0,0), thickness) 
                    if anticipated_frame:
                        # print(anticipated)
                        frame = cv2.putText(frame,"{} {}".format(class_names[anticipated_gesture], "(frame {})".format(anticipated_frame) if anticipated_gesture < 21 else ""), (120,60), font, fontScale, color, thickness) 
                        out.write(frame)
                       
                        if not showed_ant:
                            showed_ant = True
                            for _ in range(40):
                                out.write(frame)

                        # else:
                        #     cv2.waitKey(100)

                        
                    else:
                        out.write(frame)
                        # cv2.imshow("image",frame)
                        # cv2.waitKey(100)
                break

    out.release()
    print(JI_i/JI_u)


def recognition_video():
    class_names = ["no gesture","vattene","vieniqui","perfetto","furbo","cheduepalle","chevuoi","daccordo","seipazzo","combinato","freganiente","ok","cosatifarei","basta","prendere","noncenepiu","fame","tantotempo","buonissimo","messidaccordo","sonostufo","not anticipated"]
    best_samples = ["Sample0913","Sample0742","Sample0707","Sample0777"] 
    with open("results/results_anticipation_RTM_8691.pkl","rb") as file:   
        predictions = pickle.load(file)
  

    org = (20, 30) 
    org_box = (30, 30) 
    fontScale = 1
    color_bue = (200, 0, 0)
    color_green = (0,200, 0)
    color_red = ( 0, 0, 200)

    thickness = 2 
    corrects = 0
    total = 0
    JI_i,JI_u = 0,0
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('results/recognition_montalbano.avi', fourcc, 40.0, (640,480))
    for s in best_samples:
        for prediction in predictions:
            pred =  np.array(prediction["pred"])
            label = np.array(prediction["label"])
            JI_i += sum(pred==label)
            JI_u += len(label)
            probs = prediction["probs"]
            file = prediction["file"]
            acc = sum(pred==label)/len(pred)
            sample = file.split("/")[-1]
            corrects += sum(pred==label)
            acc = sum(pred==label)/len(label)
            total += len(pred)
            if sample == s:                
                print("{} -> acc {:.2f}%".format(sample,acc*100))
                video = pims.Video("samples/{}_color.mp4".format(sample))
                for frame, p, l, prob in zip(video,pred,label,probs):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.rectangle(frame, (0,0), (640,70), (200,200,200), cv2.FILLED)
                    frame = cv2.putText(frame, "Label:", (10,30), font, fontScale, (0,0,0), thickness) 
                    frame = cv2.putText(frame, "video 2x", (530,20), font, 0.8, (0,0,0)) 
                    frame = cv2.putText(frame, "Recog.:", (10,60), font, fontScale, (0,0,0), thickness) 
                    frame = cv2.putText(frame, "{}".format(class_names[l]), (130,30), font, fontScale, (0,0,0), thickness) 
                    color = color_green if p==l else color_red
                    frame = cv2.putText(frame, "{}".format(class_names[p]), (130,60), font, fontScale, color, thickness) 
                    out.write(frame)
                break

    out.release()
    print(JI_i/JI_u)




if __name__ == "__main__":
    # recognition_video()
    # calc_recognition_acc()
    # jaccard_index()

    # class_names = ["no gesture","vattene","vieniqui","perfetto","furbo","cheduepalle","chevuoi","daccordo","seipazzo","combinato","freganiente","ok","cosatifarei","basta","prendere","noncenepiu","fame","tantotempo","buonissimo","messidaccordo","sonostufo","not anticipated"]
    # # plot_hands()
    # # test()
    # # result = "results/results_dynamic_star_rgb_hand_MCLSTM_9745.pkl"
    # # with open(result,"rb") as file:   
    # #     predictions = pickle.load(file)

    # # pred, label = None, None
    # # print(len(predictions))
    # # for prediction in predictions:
    # #     # print((prediction["label"]))
    # #     # break
    # #     if pred is None:
    # #         label = prediction["label"]
    # #         pred = prediction["pred"]
    # #     else:
    # #         label = np.concatenate([label,prediction["label"]])
    # #         pred = np.concatenate([pred,prediction["pred"]])

    # # print(label.shape, pred.shape)

    # # generate_confusion_matrix(label, pred, class_names)
    # best_samples = ["Sample0913","Sample0742","Sample0707","Sample0777"] 
    # with open("results/results_anticipation_RTM_8691.pkl","rb") as file:   
    #     predictions = pickle.load(file)

    # # weigths = []
    # # labels = []
    # # preds = []
    # # for prediction in predictions:
    # #     if len(weigths)==15:
    # #         plot_only_weights(weigths, labels, preds)
    # #         plt.show()

    # #         weigths = []
    # #         labels = []
    # #         preds = []
    # #     else:
    # #         # print(prediction["label"].shape, prediction["pred"].shape)
    # #         weigths.append(prediction["weights"][:,:,0])
    # #         labels.append(prediction["label"][0])
    # #         preds.append(prediction["pred"][0])

            

    # # print(len(predictions))
  

    # org = (20, 30) 
    # org_box = (30, 30) 
    # fontScale = 1
    # color_bue = (200, 0, 0)
    # color_green = (0,200, 0)
    # color_red = ( 0, 0, 200)

    # thickness = 2 
    # corrects = 0
    # total = 0
    # JI_i,JI_u = 0,0
    # font = cv2.FONT_HERSHEY_SIMPLEX 
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('results/completo.avi', fourcc, 40.0, (640,480))
    # for s in best_samples:
    #     for prediction in predictions:
    #         pred =  np.array(prediction["pred"])
    #         label = np.array(prediction["label"])
    #         # pred = pred>0
    #         # label = label>0
    #         # pred = pred - 1
    #         # pred = split_predicition(pred,  shift = 1)
    #         # label = split_predicition(label,  shift = 0)
    #         # print(label)
    #         # print(pred)
    #         # print("*******************************************")
    #         # JI_i += sum(np.logical_and(label>0,pred==label))
    #         # JI_u +=  sum(np.logical_or(label>0,pred>0))
    #         # JI_i += levenshteinDistance(label,pred)
    #         # JI_u += len(label)

    #         # JI_i += sum((label>0) ==  (pred==label))
    #         JI_i += sum(pred==label)

    #         JI_u += len(label)


    #         # continue
            

    #         # pred = np.where(pred>0,1,0)
    #         # label = np.where(label>0,1,0)
    #         probs = prediction["probs"]
    #         weights_clf = prediction["weights_clf"]
    #         weights_spt = prediction["weights_spt"]
    #         file = prediction["file"]
    #         acc = sum(pred==label)/len(pred)
    #         sample = file.split("/")[-1]
    #         corrects += sum(pred==label)
    #         acc = sum(pred==label)/len(label)
    #         # if acc > 0.95:
    #         # print("{} -> {:.2f}".format(sample,acc*100))
    #         total += len(pred)
    #         if sample == s:
    #             plt.subplot(2, 1, 1)
    #             plot_weights(weights_spt, label,pred, "Spotting")
    #             plt.subplot(2, 1, 2)
    #             plot_weights(weights_clf, label,pred,"Classificação")
    #             # plt.title("Pesos do soft-attention por amostra")
    #             plt.show()
    #             continue

                
    #             print("{} -> acc {:.2f}%".format(sample,acc*100))
    #             video = pims.Video("samples/{}_color.mp4".format(sample))
    #             # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #             # out = cv2.VideoWriter('results/{}.avi'.format(sample), fourcc, 20.0, (640,480))
    #             label_before = 0
    #             count = 0
    #             anticipated_frame = False
    #             anticipated_gesture = False
    #             showed_ant = False
    #             for frame, p, l, prob in zip(video,pred,label,probs):
    #                 _,_,mi = calc_uncertainties(prob)
    #                 if label_before == 0 and l>0:
    #                     label_before = 1
    #                     count = 1
    #                     anticipated_frame= False
    #                     anticipated_gesture = False
    #                     showed_ant = False
    #                 elif label_before == 1 and l==0:
    #                     label_before = 0
    #                     count = 0
    #                     if not showed_ant:

    #                             anticipated_frame = 1
    #                             anticipated_gesture = 21
    #                             color =  color_red
    #                     else:
    #                         anticipated_gesture = False
    #                         anticipated_frame = False
    #                     showed_ant = False
    #                 elif l >0:
    #                         count += 1
    #                         if (p>0) and (mi<=0.2) and (not anticipated_frame):
    #                             showed_ant = False
    #                             anticipated_frame = count
    #                             anticipated_gesture = p
    #                             color = color_green if p==l else color_red
    #                 elif l == 0 and anticipated_gesture == 21:
    #                     label_before = 0
    #                     count = 0
    #                     anticipated_gesture = False
    #                     anticipated_frame = False
    #                     showed_ant = False
                            

                    

    #                 frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #                 frame = cv2.rectangle(frame, (0,0), (640,70), (200,200,200), cv2.FILLED)
    #                 frame = cv2.putText(frame, "Label:", (10,30), font, fontScale, (0,0,0), thickness) 
    #                 frame = cv2.putText(frame, "video 2x", (530,20), font, 0.8, (0,0,0)) 
    #                 frame = cv2.putText(frame, "Antic.:", (10,60), font, fontScale, (0,0,0), thickness) 
    #                 frame = cv2.putText(frame, "u=0.2", (530,50), font, 0.8, (0,0,0))
    #                 frame = cv2.putText(frame, "{}".format(class_names[l]), (120,30), font, fontScale, (0,0,0), thickness) 
    #                 if anticipated_frame:
    #                     # print(anticipated)
    #                     frame = cv2.putText(frame,"{} {}".format(class_names[anticipated_gesture], "(frame {})".format(anticipated_frame) if anticipated_gesture < 21 else ""), (120,60), font, fontScale, color, thickness) 
    #                     out.write(frame)
                       
    #                     if not showed_ant:
    #                         showed_ant = True
    #                         for _ in range(40):
    #                             out.write(frame)

    #                     # else:
    #                     #     cv2.waitKey(100)

                        
    #                 else:
    #                     out.write(frame)
    #                     # cv2.imshow("image",frame)
    #                     # cv2.waitKey(100)
    #             break

    # out.release()
    #         # plt.plot(label)
    #         # plt.plot(pred)
    #         # plt.legend(["label","pred"])
    #         # plt.show()
    #         #break
    # # print(corrects/total)
    # print(JI_i/JI_u)







