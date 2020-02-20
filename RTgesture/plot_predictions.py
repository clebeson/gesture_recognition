import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import product
import pickle
import pims
import cv2
import matplotlib as mpl
# from context import draw_ballpose, draw_bodypose, draw_class, draw_hand_center, draw_handpose, draw_head
#mpl.use('Agg')
mpl.style.use("seaborn")

def plot_probabilities(probs,pred, label):
    plt.plot(probs)
    plt.legend(['act {}'.format(i) for i in range(1,13)], loc='upper right')
    plt.title("Label = {} / Predicted = {}".format(label+1,pred+1))

#def plot_conf_matrix(predictions, labels): pass

def anticipate(probabilities, th = 0.9, uncert = True):
    if uncert:
        _,_,mi= calc_uncertainties(probabilities)
        if mi < th: return (True, probabilities.mean(0).argmax(),probabilities.mean(0).max(),mi)
        return (False, -1, probabilities.mean(0).max(), mi)
    else: 
        if probabilities.max() > th:
            return (True, probabilities.argmax(),probabilities.max(), 0)
        return (False, -1, 0, probabilities.max())




def generate_video():
    stochastic = pickle.load(open("results/mc_dropout.pkl", 'rb'), encoding="bytes")
    deterministic = pickle.load(open("results/prediction_m_g_b.pkl", 'rb'), encoding="bytes")
    stochastic.sort(key = lambda d:d[b'interval'][0])
    deterministic.sort(key = lambda d:d[b'interval'][0])
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('selected.avi',fourcc, 15.0, (1280,480))
    chosen = [2988, 6155,1476, 2096, 6394, 30,74, 6210, 1562, 2178, 1108,6442, 6499,8686,9152, 13919, 10228]

    

    actions = {
                1:"Ask",
                2:"Come_Here",
                3:"Leave",
                4:"Follow",
                5:"Stop",
                6:"Abort",
                7:"Good",
                8:"No",
                9:"Bad",
                10:"Give_way",
                11:"Pointing",
                12:"Doubt",
                13:"Louder",
                14:"Quieter",
                15:"Be_quiet"
        }
    colors = ["green", "black", "red", "blue", "brown", "indigo","coral","lime","orangered","yellow", "navy","salmon","gray","darkorange","deepskyblue"]
    font = font_manager.FontProperties(weight='normal',
                                   style='normal', size=10)
    font_act = font_manager.FontProperties(weight='bold',
                                   style='normal', size=10)
    box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    frame_count = -1
    for det, stoc in zip(deterministic, stochastic):
        label = stoc[b'label']
        det_prob = det[b'probs']
        stoc_prob = stoc[b'probs']
        begin,end= stoc[b'interval']
        if begin not in chosen: continue
        mean = stoc_prob.mean(1)
        std = stoc_prob.std(1)
        std_up  = np.clip(mean+std,0.0,1.0)
        std_down = np.clip(mean-std,0.0,1.0)
        indexes = np.arange(len(mean))
        lines=[]
        print(indexes.shape,std_up.shape,std_down.shape)
        det_ant = -1
        stoc_ant = -1
        act_s = -1
        act_s = -1
        frame_count = begin-1
        for i in range(len(mean)):
            frame_count += 1
            fig, axis = plt.subplots(2,1,facecolor=(0.8, 0.8, 0.8))
            axis[0].set_xlim([0,len(det_prob)])
            axis[1].set_xlim([0,len(stoc_prob)])
            axis[0].set_ylim([0.0,1.0])
            axis[1].set_ylim([0.0,1.0])
            axis[0].set_xlabel("Frame",  weight="bold")
            axis[1].set_xlabel("Frame",  weight="bold")
            axis[0].set_ylabel("Probability",  weight="bold")
            axis[1].set_ylabel("Probability",  weight="bold")
            plt.subplots_adjust(hspace=1.2)
            axis[1].plot([0,len(det_prob)],[0.9,0.9],linestyle='-',color="b", linewidth=1) 
            fig.suptitle("Action {} - ({})".format(label+1,actions[label+1]), fontsize=15, weight="bold")

            for v in range(12): 
                line, = axis[0].plot(mean[:i,v], color=colors[v])
                axis[0].fill_between(indexes[:i],std_up[:i,v], std_down[:i,v], alpha=0.3, facecolor=colors[v])
                axis[1].plot(det_prob[:i,v], color=colors[v])
                if len(lines) < len(colors):lines.append(line)
            if det_ant == -1:
                ant_d,act_d,prob_d,_ = anticipate(det_prob[i],0.9,False)
            if stoc_ant == -1:
                ant_s,act_s, prob_s, uncertainty = anticipate(stoc_prob[i],0.5,True)
            else: _,_, _, uncertainty = anticipate(stoc_prob[i],0.5,True)
            
            
            if ant_s or stoc_ant> -1:
                if stoc_ant == -1:
                    stoc_ant = i
                plot_anticipation(axis[0], prob_s, stoc_ant)
                tx,ty = [2, 0.83] if stoc_ant>0.5 else [stoc_ant+3,0.83]
                axis[0].text(2,0.75 ,  "    Act. {} at Frame {}".format(act_s+1,stoc_ant+1),fontproperties = font_act, bbox = box)
                if act_s == label:
                    axis[0].text(2,0.75 , "V",bbox = box, fontproperties = font_act,color = "g")
                else:
                    axis[0].text(2,0.75 , "X",bbox = box, fontproperties = font_act,color = "r")
            
            if ant_d or det_ant>-1:
                if det_ant == -1:det_ant = i
                plot_anticipation(axis[1], prob_d, det_ant)
                tx,ty = [2, 0.83] if det_ant>0.5 else [det_ant+3,0.83]
                axis[1].text(2,0.75 ,  "    Act. {} at Frame {}".format(act_d+1,det_ant+1),fontproperties = font_act, bbox = box)
                if act_d == label:
                    axis[1].text(2,0.75 , "V",bbox = box, fontproperties = font_act,color = "g")
                else:
                    axis[1].text(2,0.75 , "X",bbox = box, fontproperties = font_act,color = "r")
            
            axis[0].set_title("Stochastic Model - Uncertainty = {:.2f}".format(uncertainty),  weight="bold")
            axis[1].set_title("Deterministic Model",  weight="bold")


            fig.legend(lines, ["{} - {}".format(i,actions[i]) for i in range(1,len(colors)+1)], loc='center', \
        prop = font, ncol = 4, shadow=True,  frameon=True, fancybox=True)

            frame = framesw[frame_count]
            # plt.legend(['act {}'.format(i) for i in range(1,probs.shape[1]+1)], loc='upper right')
            # #print(entropy(probs[i]))
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            img_plot = cv2.resize(img, (640, 480))
            image = np.concatenate((img_plot,frame),axis=1)
            out.write(image[:,:,[2,1,0]])
            # cv2.imshow("Action Dataset",image[:,:,[2,1,0]])
            # cv2.waitKey(10)
            plt.close()
            





def plot_anticipation(axis, acc,frame):
    axis.plot([0,frame],[acc,acc],linestyle='--',color="k", linewidth=1)
    axis.plot([frame,frame],[0,acc],linestyle='--',color="k", linewidth=1)
    




def show_video(frames, probs,probs_u, begin,end):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_{}.avi'.format(int(begin)),fourcc, 5.0, (1280,480))
        
        #labels_body = np.load("labels_body.npy").astype(int)
        fig, axis = plt.subplots(2,1,facecolor=(0.6843, 0.9098, 0.9098))
        axis[0].set_xlim([0,len(probs)])
        axis[1].set_xlim([0,len(probs)])
        axis[0].set_ylim([0.0,1.0])
        axis[1].set_ylim([0.0,1.0])
        axis[0].set_xlabel("Frame")
        axis[1].set_xlabel("Frame")
        axis[0].set_ylabel("Probability")
        axis[1].set_ylabel("Probability")

        draw_hand = True
        draw_body = True
        draw_ball = True
        draw_gaze = True

        _,_,mi = calc_uncertainties(probs_u)
        meanst = probs_u.mean(1)
        std = probs_u.std(1)
        y1  = np.clip(meanst-std,0.0,1.0)
        y2 = np.clip(meanst+std,0.0,1.0)

        empty = False
        empty_image = (np.ones((480,640,3))*127).astype(np.uint8) 
        framesw = pims.Video("./dataset/world.mp4")
        labels = np.load("labels_complete_cut.npy").astype(int) 
        #cv2.imshow("Action Dataset",empty_image)
        labels = labels[begin:end]
        colors = ["green", "black", "red", "blue", "brown", "indigo","coral","lime","orangered","yellow", "navy","salmon"]
        indexes = np.arange(len(meanst))

        for i, label in enumerate(labels):
                
                # convert canvas to image
             
                frame = frames[label[0]]
                cv2.imwrite('start.png',frame[:,:,[2,1,0]])
        
                ball = label[2:4]
                body = label[4:40].reshape((18,2)).astype(int)
                hand = label[40:44].reshape((2,2)).astype(int)


                frame = empty_image #if empty else frame
                frame = draw_bodypose(frame,body) if draw_body else frame
                frame = draw_head(frame,body) if draw_gaze else frame
                frame = draw_hand_center(frame,hand) if draw_hand else frame
                frame = draw_ballpose(frame,ball.astype(int)) if draw_ball else frame
                cv2.imwrite('skeleton.png',frame[:,:,[2,1,0]])
                #frame = draw_class(frame, label[1])



                # for j in range(probs.shape[1]):
                #     axis[0].plot(probs[:i,j], color = colors[j])
                #     axis[1].plot(meanst[:i,j], color=colors[j])
                #     axis[1].fill_between(indexes[:i],y1[:i,j], y2[:i,j],alpha=0.1, facecolor=colors[j])
                # plt.legend(['act {}'.format(i) for i in range(1,probs.shape[1]+1)], loc='upper right')
                # #print(entropy(probs[i]))
                # fig.canvas.draw()

                # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
                # img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                # img_plot = cv2.resize(img, (640, 480))
                

                # image = np.concatenate((img_plot,frame),axis=1)
                #out.write(image[:,:,[2,1,0]])
                cv2.imshow("Action Dataset",frame[:,:,[2,1,0]])
                k = cv2.waitKey(3000)
                if k == 27:break
                if k == 98:draw_ball = not draw_ball #B
                if k == 103:draw_gaze = not draw_gaze #G
                if k == 101:empty = not empty #E
                if k == 104:draw_hand = not draw_hand #H
                if k == 106:draw_body = not draw_body #J
                if k == 32: 
                        while cv2.waitKey(30) != 32:continue
        plt.close()
        out.release()


def plot_all_charts():
    #
    #titles = ["$DLSTM_{12m}$ (Movement)", "$DLSTM_{12h}$ (Head)","$DLSTM_{12o}$ (Object)","$DLSTM_{12mh}$ (Movement + Head)", "$DLSTM_{12mo}$ (Movement + Object)","$DLSTM_{12mho}$ (Movement+Head+Object)"]
    titles = ["$DLSTM_{12m}$ ", "$DLSTM_{12h}$ ","$DLSTM_{12o}$ ","$DLSTM_{12mh}$  ", "$DLSTM_{12mo}$ ","$DLSTM_{12mho}$ "]
   
    colors = ["green", "black", "blue", "red","brown", "indigo","coral","lime","orange","yellow", "navy","salmon","gray","darkgray","darkorange"]
    
    #titles = ["Movement", "Head","Movement + Head"]
    #colors = ["green", "black", "blue", "red","brown", "indigo"]

    actions = {
                1:"Ask",
                2:"Come_Here",
                3:"Leave",
                4:"Follow",
                5:"Stop",
                6:"Abort",
                7:"Good",
                8:"No",
                9:"Bad",
                10:"Give_way",
                11:"Pointing",
                12:"Doubt",
                13:"Louder",
                14:"Quieter",
                15:"Be_quiet"
        }
    

    videos= pickle.load(open("prediction_gesture3_0_91.16.pkl", 'rb'), encoding="bytes")


    font_act = font_manager.FontProperties(weight='bold',
                                   style='normal', size=10)
    box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for video in videos:
        
        fig, axis = plt.subplots(1,1, figsize=(18, 10))
        
        fig.subplots_adjust(hspace=0.77, left = 0.04, right = 0.99, bottom = 0.1,top=0.9)
        lines = []
        
        axis.set_ylim(0.0,1.0)
        axis.set_xlabel("Observation Ratio", fontsize=14)
        axis.set_ylabel("Probability", fontsize=14)
        #a.tick_params(labelsize=6)
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
            

        
        begin,end = video['interval']
        probs = video['probs'].mean(1)
        label = video['label']
        pred = video['pred']
        print(probs.shape)
        p = np.amax(probs, axis= 1)
        c = np.argmax(probs, axis= 1)
        x = np.argmax(p>=0.9, axis=0)
        print(x)
        y = p[x]
        a = c[x]
            
        #ratio = len(probs)/100.0
        #probs = np.array([probs[int(i*ratio)] for i in range(100)])
        #observation = [i/100.0  for i in range(100)]
        total = len(probs)
        print(total)
        x = x/float(total)
        observation = [i / float(total) for i in range(total)]
        #axs[i].set_xlim(0,len(probs))
        for cl, color in enumerate(colors):
            line = axis.plot(observation, probs[:,cl], color = color)
            if len(lines) < len(colors):lines.append(line)
        #axis.legend([l for l in range(1,13)], loc='upper right',fontsize = 4, prop = {'weight':'bold'})
        
        line_k = axis.plot([x,x,],[0,y],linestyle='--',color="k", linewidth=3)
        axis.plot([0,x,],[y,y],linestyle='--',color="k", linewidth=3)
        axis.plot(x, y, color='green', linestyle='dashed', marker='o', markerfacecolor='k', markersize=10)
        
        if y >= 0.9:
            tx,ty = [0.02, 0.83] if x>0.5 else [x+0.03,0.83]
            axis.text(0.2,0.55 ,  "    Act. {} with {}% of Frames".format(a+1,round(x*100)),fontproperties = font_act, bbox = box)
            if a == label:
                axis.text(0.2,0.55 , "V",bbox = box, fontproperties = font_act,color = "g")
            else:
                axis.text(0.2,0.55 , "X",bbox = box, fontproperties = font_act,color = "r")
        else:
            axis.plot([0,1.0,],[0.9,0.9],linestyle='--',color="k", linewidth=3)
            axis.text(0.2,0.55 , "It was not possible to anticipate", bbox = box,fontproperties = font_act,color = "r")
        

        axis.set_title("{} - Recognized({})".format(titles[i],pred+1), weight="normal")
        
        
        fig.suptitle('{} - {}'.format(actions[label+1],label+1), fontsize=20, weight="normal")
        
        font = font_manager.FontProperties(weight='normal',
                                   style='normal', size=13)
        legend = fig.legend(lines+[line_k], ["{} - {}".format(i,actions[i]) for i in range(1,len(colors)+1)]+["$p=0.9$"], loc='center', \
        prop = font, ncol = 7, shadow=True,  frameon=True, fancybox=True)
        #legend.get_frame().set_facecolor((1.0,1.0,1.0))
        
        plt.show()
        return
        plt.savefig('charts/chart_act_({}-{})_{}_{}.png'.format(begin,end,label+1,v))
        plt.close()
        #return

def plot_uncertainty_threshold():

    
    
    # values = pickle.load(open("results/prediction_brnn_6441.pkl", 'rb'), encoding="bytes")
    # values.sort(key = lambda d:d['interval'][0])
    # values_u = values
    
    # for i,(value,value_u) in enumerate(zip(values,values_u)):
    #     label = value_u['label']
    #     pred = value_u['pred']
    #     begin,end = value_u['interval']
    #     probs = value['probs']
    #     probs_u = value_u['probs']
    #     print(begin,end)
    #     if label == pred:
    #         frames = pims.Video("./dataset/external.mp4")
    #         show_video(frames,probs,probs_u,begin,end)
            
    #         return
        
        
       
        
    values_u = pickle.load(open("prediction_gesture3_0_91.16.pkl", 'rb'), encoding="bytes")
    #interval = []
    #pred = []
    #probs = []
    #label = []
    #for value in values_u:
    #    interval.append(value['interval'])
    #    pred.append(value[b'pred'])
    #    probs.append(value[b'probs'])
    #    label.append(value[b'label'])
    #values_u = { 'interval':np.array(interval),'pred':np.array(pred),'probs':np.array(probs),'label':np.array(label)}
    values_u.sort(key = lambda d:d['interval'][0])
    results = []
    for t in range(50):
        predictions = {"classes":[], "labels":[]}   
        t /= 10.0
        corrects = []
        total_frames = []
        corrects_ant = 0
        anticipate = []
        for i,value in enumerate(values_u):
            label = value['label']
            pred = value['pred']
            begin,end = value['interval']
            probs = value['probs']

            vr,h,mi = calc_uncertainties(probs)
            meanst = probs.mean(1)
            std = probs.std(1)
            c = np.argmax(meanst, axis= 1)

            x = np.argmax(mi<t)
            a = c[x]
            if x >0 and a == label:
                corrects_ant += 1.0
            # else: 
            #     # print("x {}, begin {}, pred {}, label {}".format(x,begin,a,label))
            #     print(begin)
            # if x == 0:a = 0

            # predictions["labels"].append(label)
            # predictions["classes"].append(a)

            ant = float(x)/len(probs) if  x > 0 else 1.0 #len(probs)
            anticipate.append(ant)
            total_frames.append(len(probs))

            corrects.append(pred==label)
        # plot_conf_matrix(predictions)
        # return
        acc = corrects_ant/len(corrects)
        m = sum(total_frames)/len(total_frames)
        ant = (sum(anticipate)/len(anticipate))
        results.append([t,acc,ant])
    
    
    results = np.array(results)   
    #results[:,2]/=100
    
    acc = results[:,1].max()
    pos = results[:,1].argmax()
    u_acc = results[pos,0]
    acc_frame =  results[pos,2]
    
    frame = results[:,2].min()
    pos = results[:,2].argmin()
    u_frame = results[pos,0]
    frame_acc =  results[pos,1]

    print(acc, frame, u_acc, u_frame)
    fig, axis = plt.subplots(1,1,figsize=(12, 5))
    g,=axis.plot(results[:,0],results[:,1],color = "g")
    o, = axis.plot(results[:,0],results[:,2],color = "orange")
    k,= axis.plot([u_acc,u_acc],[0,acc],linestyle='--',color="k", linewidth=1)
    axis.plot([0,u_acc],[acc,acc],linestyle='--',color="k", linewidth=1)
    axis.plot([0,u_acc],[acc_frame,acc_frame],linestyle='--',color="k", linewidth=1)
    
    axis.text(u_acc,acc+0.01,"{:.2f}%".format(acc*100))
    axis.text(u_frame,frame_acc+0.01,"{:.2f}%".format(frame_acc*100))

    b,=axis.plot([u_frame,u_frame],[0,frame],linestyle='-.',color="b", linewidth=1)
    axis.plot([0,u_frame],[frame,frame],linestyle='-.',color="b", linewidth=1)
    axis.plot([0,u_frame],[frame_acc,frame_acc],linestyle='-.',color="b", linewidth=1)
    axis.plot([u_frame, u_frame],[frame_acc,frame],linestyle='-.',color="b", linewidth=1)
    
    axis.text(u_acc+0.01, acc_frame+0.01,"{}% of Frames".format(int(acc_frame*100)))
    axis.text(u_frame+0.01,frame-0.04,"{}% of Frames".format(int(frame*100)))


    axis.set_xlabel("Uncertainty (Mutual Information)")
    axis.set_ylabel("Anticipation Accuracy / Observation Ratio(OR)")
    axis.legend([g,o,k,b],["Anticipation Accuracy", "Average OR Anticipation","Maximum Anticipation Accuracy","Minimum Average OR Anticipation"],loc="center right",framealpha=1, frameon=True, fancybox=True)
    axis.set_title("Anticipation vs Uncertainty ($BLSTM_{MC}$)", weight="bold")
    plt.show()



def generate_confusion_matrix( predictions, class_names):
        
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
                print(cm)
                print(accuracy_score(predictions["labels"], predictions["classes"]))
                

                plt.imshow(cm, interpolation='nearest', cmap=cmap)
                # plt.title(title)
                plt.colorbar()
            
        
                plt.xticks(list(range(len(class_names))), classes, rotation=45)
                plt.yticks(list(range(len(class_names))), classes)

                fmt = '.1f' if normalize else 'd'
                thresh = cm.max() / 2.
                symbol = "%" if normalize else ""
                for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                    
                    if cm[i, j] > 0:
                        #if i == j:
                            plt.text(j, i, format(cm[i, j], fmt),
                                    fontsize=12, ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                # plt.ylabel('Real')
                # plt.xlabel('Predicted')
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(predictions["labels"],predictions["classes"])
        np.set_printoptions(precision=2)
        

        # # Plot normalized confusion matrix
        plt.figure(figsize=(8,8))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            # title='Normalized confusion matrix'
                            )
        plt.grid(None)
        plt.show()
        # plt.savefig("conf_DLSTM_complete.svg", format="svg")
        

def plot_conf_matrix(predictions = None):
    if predictions is None:
        values= pickle.load(open("prediction_gesture1_55.22.pkl", 'rb'), encoding="bytes")
        
        predictions = {"classes":[], "labels":[]}
    
        for i,value in enumerate(values):
            predictions["labels"].append(value['label'])
            predictions["classes"].append(value['pred'])
    
    names = ["Ask","Come_Here","Leave","Follow","Stop","Abort","Good","No","Bad","Give_way","Pointing","Doubt","Louder","Quieter","Be_quiet"]
    generate_confusion_matrix(predictions,names)
    


def plot_probability_threshold():
    values= pickle.load(open("results/prediction_m_g_b.pkl", 'rb'), encoding="bytes")
    predictions = {"classes":[], "labels":[]}
    results = []
    for t in range(100):
        t /= 100.0
        corrects = []
        total_frames = []
        corrects_ant = 0
        anticipate = []
        anticipated = -1
        for i,value in enumerate(values):
            label = value[b'label']
            pred = value[b'pred']
            begin,end = value[b'interval']
            probs = value[b'probs']

            c = np.argmax(probs, axis= 1)
            p = np.amax(probs, axis= 1)
            
            m = p>t
            x = np.argmax(m)
            count = 0

            # if x>0:
            #     for j in range(x, len(m)):
            #         if m[j]:
            #             count += 1
            #             x = j
            #         else:
            #             count = 0
            #             x = 0
                        
            #         if count == t:break
                    
            a = c[x]
            if x >0 and a == label:
                corrects_ant += 1.0
            
            #predictions["labels"].append(label)
            #predictions["classes"].append(a)

            ant = float(x)/len(probs) if  x > 0 else 1.0 #len(probs)
            anticipate.append(ant)
            total_frames.append(len(probs))

            corrects.append(pred==label)
        #plot_conf_matrix(predictions)
        #return
        acc = corrects_ant/len(corrects)
        m = sum(total_frames)/len(total_frames)
        ant = (sum(anticipate)/len(anticipate))
        
        results.append([t,acc,ant])
    
    results = np.array(results)   
   #
    results[0,[1,2]] = results[1,[1,2]]
    
    acc = results[:,1].max()
    pos = results[:,1].argmax()
    u_acc = results[pos,0]
    acc_frame =  results[pos,2]
    
    frame = results[:,2].min()
    pos = results[:,2].argmin()
    u_frame = results[pos,0]
    frame_acc =  results[pos,1]

    print(acc, frame, u_acc, u_frame)
    fig, axis = plt.subplots(1,1,figsize=(12, 5))
    g,=axis.plot(results[:,0],results[:,1],color = "g")
    o, = axis.plot(results[:,0],results[:,2],color = "orange")
    k,= axis.plot([u_acc,u_acc],[0,acc],linestyle='--',color="k", linewidth=2)
    axis.plot([0,u_acc],[acc,acc],linestyle='--',color="k", linewidth=2)
    axis.plot([0,u_acc],[acc_frame,acc_frame],linestyle='--',color="k", linewidth=2)
    
    axis.text(u_frame,frame_acc-0.05,"{:.2f}%".format(frame_acc*100))
    axis.text(u_acc,acc+0.01,"{:.2f}%".format(acc*100))

    b,=axis.plot([u_frame,u_frame],[0,frame],linestyle='-.',color="b", linewidth=1)
    axis.plot([0,u_frame],[frame,frame],linestyle='-.',color="b", linewidth=1)
    axis.plot([0,u_frame],[frame_acc,frame_acc],linestyle='-.',color="b", linewidth=1)
    axis.plot([u_frame, u_frame],[frame_acc,frame],linestyle='-.',color="b", linewidth=1)
    

    axis.text(u_acc+0.01, acc_frame+0.01,"{}%".format(int(acc_frame*100)))
    axis.text(u_frame+0.01,frame-0.04,"{}%".format(int(frame*100)))

    axis.set_ylabel("Anticipation Accuracy / Observation Ratio(OR)")

    axis.set_xlabel("Probability")
    axis.legend([g,o,k,b],["Anticipation Accuracy", "Average OR Anticipation","Maximum Anticipation Accuracy","Minimum Average OR Anticipation"],loc="center left", framealpha=1, frameon=True)
    axis.set_title("Anticipation vs Probability ($DLSTM_{mho}$)", weight="bold")
    
    # axis.set_xlabel("Aditional OR after the first prediction (z)")
    # axis.legend([g,o,k],["Anticipation Accuracy (threshold = 0.9)", "Average OR Anticipation","Maximum Anticipation Accuracy"],loc="center right", framealpha=1, frameon=True)
    #axis.set_title("Anticipation with threshold = 0.9", weight="bold")
    plt.show()



def plot_uncertainty():
    font_act = font_manager.FontProperties(weight='bold',
                                   style='normal', size=10)
    box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    th=0.2
    values = pickle.load(open("prediction_gesture3_0_91.16.pkl", 'rb')) #prediction_BBB_9833.pkl
    values.sort(key = lambda d:d['interval'][0])
    colors = ["green", "black", "blue", "red","brown", "indigo","coral","lime","orange","yellow", "navy","salmon","gray","darkorange","deepskyblue"]
    for i,value in enumerate(values):
        lines = []
        label = value['label']
        pred = value['pred']
        b,e  = value['interval'] 
        fig, axs = plt.subplots(1, 2,figsize=(20, 5))
        begin,end = value['interval']
        probs = value['probs']
        vr,h,mi = calc_uncertainties(probs)
        meanst = probs.mean(1)
        std = probs.std(1)

        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("Probability")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Mutual Information")

        y1  = np.clip(meanst-std,0.0,1.0)
        y2 = np.clip(meanst+std,0.0,1.0)
        p = np.amax(meanst, axis= 1)
        c = np.argmax(meanst, axis= 1)
        std = std[range(len(c)),c]
        xu = np.argmax(mi < th) #using uncertainty
        x = np.argmax(p>=0.9)
        #print(std[xu])

        y = p[x]
        yu = p[xu]
        #hc = h[x]
        mic = mi[x]
        #hu = h[xu]
        miu = mi[xu]
        a = c[xu]
        indexes = np.arange(len(meanst))

        #axs[1].plot(norm(vr))
        #axs[1].plot(h )
        line_mi, = axs[1].plot(mi,color="g")
        line_h, = axs[1].plot(h,color="r")
        line_vr, = axs[1].plot(vr,color="orange")
        

        axs[0].set_xlim(0,len(probs)+10)
        line_k, line_b = None, None
        for v in range(15): 
             line, = axs[0].plot(meanst[:,v], color=colors[v])
             axs[0].fill_between(indexes,y1[:,v], y2[:,v],alpha=0.3, facecolor=colors[v])
             if len(lines) < len(colors):lines.append(line)

        if xu > 0:
            axs[0].plot([xu,xu],[0,yu],linestyle='--',color="k", linewidth=2)
            axs[0].plot([0,xu],[yu,yu],linestyle='--',color="k", linewidth=2)

            line_k,=axs[1].plot([xu,xu],[0,miu],linestyle='--',color="k", linewidth=2)
            axs[1].plot([0,xu],[miu,miu],linestyle='--',color="k", linewidth=2)
            
            tx,ty = [2,yu-0.07] if xu>70 else [xu+3,yu-0.07]
            axs[0].text(tx,ty ,  "    Act. {} at Frame {}".format(a+1,xu),fontproperties = font_act, bbox = box)
            if a == label:
                 axs[0].text(tx,ty , "V",bbox = box, fontproperties = font_act,color = "g")
            else:
                axs[0].text(tx,ty , "X",bbox = box, fontproperties = font_act,color = "r")
                #plt.savefig('charts_uncertainty/chart_act{}_{}.png'.format(str(label+1),i))
        else:
            axs[0].text(10, 0.5 , "It was not possible to anticipate", bbox = box,fontproperties = font_act,color = "r")
            line_k,=axs[1].plot([0,len(mi)],[0.1,0.1],linestyle='--',color="k", linewidth=2)

        if x>0:
            axs[0].plot([x,x,],[0,y],linestyle='-.',color="b", linewidth=2)
            axs[0].plot([0,x,],[y,y],linestyle='-.',color="b", linewidth=2)

            axs[1].plot([x,x,],[0,mic],linestyle='-.',color="b", linewidth=2)
            line_b, =axs[1].plot([0,x,],[mic,mic],linestyle='-.',color="b", linewidth=2)
       
                
        axs[0].set_title("label ({}) / Prediction ({})".format(label+1,pred+1), weight="bold")
        axs[0].legend(lines+[line_k,line_b],["{}".format(i) for i in range(1,len(colors)+1)]+["MI < {}".format(th),"prob >= 0.9"],loc="center right")
        axs[1].legend([line_mi,line_h,line_vr,line_k,line_b],["MI","MI < {}".format(th),"prob >= 0.9"])
        axs[1].set_title("Uncertainty - (MI - Mutual Information)", weight="bold")
        #plt.show()
        #return 
        #if xu<=0 or a != label:

        plt.savefig('charts/chart_act_({}-{})_{}_{}.png'.format(begin,end,label+1,i))
        plt.close()



def show_many(frames, m,g,b,mg,mgb, out):
        #labels_body = np.load("labels_body.npy").astype(int)

        probs  = np.array([m[b'probs'], g[b'probs'], b[b'probs'],mg[b'probs'] , mgb[b'probs']])
       #pred = np.array([m[b'pred'], mg[b'pred'] , mgb[b'pred']])
       # label = np.array([m[b'label'], mg[b'label'] , mgb[b'label']])
        begin,end = m[b'interval']
        
        


        fig, axs = plt.subplots(2, 3)
        axs = axs.reshape((-1,))
        fig.subplots_adjust(hspace=0.5)
        titles = ["Movement", "Gaze","Ball","Movement + Gaze ", "Movement + Gaze + Ball",""]
        for a,t in zip(axs,titles):
            if t =="":
                a.grid(False)
                a.set_yticklabels([])
                a.set_xticklabels([])
            else:
                a.set_xlim(0,len(probs[0]))
                a.set_ylim(0.0,1.0)
                a.set_xlabel("Frame")
                a.set_title(t)
                a.set_ylabel("Accuracy")
                a.tick_params(labelsize=6)
        
        # fig = plt.figure(facecolor=(0.6843, 0.9098, 0.9098))
        # plt.xlim([0,len(prob_m)])
        # plt.ylim([0.0,1.0])
        # plt.xlabel("Frame")
        # plt.ylabel("Accuracy")

        draw_hand = True
        draw_body = True
        draw_ball = True
        draw_head = True
        empty = False
        empty_image = (np.ones((480,640,3))*127).astype(np.uint8) 
        #framesw = pims.Video("./dataset/world.mp4")
        labels = np.load("labels_complete_cut.npy").astype(int) 
        #cv2.imshow("Action Dataset",empty_image)
        labels = labels[begin:end]
        colors = ["green", "black", "blue", "red","brown", "indigo","coral","lime","orange","yellow", "navy","salmon"]
        for i, label in enumerate(labels):

                # convert canvas to image

                frame = frames[label[0]]
                ball = label[2:4]
                body = label[4:40].reshape((18,2)).astype(int)
                hand = label[40:44].reshape((2,2)).astype(int)


                frame = empty_image if empty else frame
                frame = draw_head(frame,body) if draw_head else frame
                frame = draw_bodypose(frame,body) if draw_body else frame
                frame = draw_hand_center(frame,hand) if draw_hand else frame
                frame = draw_ballpose(frame,ball.astype(int)) if draw_ball else frame
                frame = draw_class(frame, label[1])
                #frame = cv2.resize(frame, (320, 240))


                for a,p in zip(axs[:-1],probs):
                    for j in range(12):
                        a.plot(p[:i,j], color = colors[j])
                        a.legend([i for i in range(1,13)], loc='upper right',fontsize = 6)
                axs[-1].imshow(frame)
                fig.canvas.draw()

                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                img_plot = cv2.resize(img, (1280, 720))
                img_plot = img_plot[:,:,[2,1,0]]
                out.write(img_plot)
                continue

                #image = np.concatenate((img_plot,frame),axis=1)
                cv2.imshow("Action Dataset",img_plot[:,:,[2,1,0]])
                k = cv2.waitKey(10)
                print(k)
                if k == 27:break
                if k == 98:draw_ball = not draw_ball #B
                if k == 101:empty = not empty #E
                if k == 104:draw_hand = not draw_hand #H
                if k == 106:draw_body = not draw_body #J
                if k == 32: 
                        while cv2.waitKey(30) != 32:continue
        plt.close()

def plot_acc_classes():
    th_acc = [0.8,0.9]
    th_ent = [0.6,0.38]
    #titles_acc = ["Movement", "Head", "Object", "Movement + Head", "Movement + Object", "Movement + Head + Object","Threshold at {}".format(th_acc[0]), "Threshold at {}".format(th_acc[1])]
    titles_acc = ["$DLSTM_{6m}$ (Movement)", "$DLSTM_{6h}$ (Head)", "$DLSTM_{6o}$ (Object)", "$DLSTM_{6mh}$ (Movement + Head)",  "$DLSTM_{6mo}$ (Movement + Object)",  "$DLSTM_{6mho}$ (Movement + Head + Object)",  "{}% of accuracy".format(int(th_acc[0]*100)), "{}% of accuracy".format(int(th_acc[1]*100))]
    titles_ent = ["Movement", "Head", "Object", "Movement + Head", "Movement + Object", "Movement + Head + Object","Threshold at {}".format(th_ent[0]), "Threshold at {}".format(th_ent[1])]
    #titles_ent = ["Movement", "Head", "Movement + Head",  "Threshold at {}".format(th_ent[0]), "Threshold at {}".format(th_ent[1])]

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    actions = [0,1,2,3,4,5]
    actions = [6,7,8,9,10,11]
    actions = [0,1,2,3,4,5,6,7,8,9,10,11]
    acc_classes = np.zeros((100,len(titles_acc)-2))
    entropies = np.zeros((100,len(titles_acc)-2))
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    fig.subplots_adjust(hspace=1.0)
    #axs = axs.reshape((-1,))
    for s, source in enumerate( ["mov","gaze","ball","mov_gaze", "mov_ball", "complete"]):
    #values = pickle.load(open("results/prediction_m_g_b.pkl", 'rb'), encoding="bytes")
    #values = pickle.load(open("./results/prediction_6_g.pkl", 'rb'), encoding="bytes")
    # if len(sys.argv) >1:
    #     source = sys.argv[1]
        values = pickle.load(open("./results/prediction_m_g_b.pkl", 'rb'),encoding="bytes")
    
        if source =="gaze":
            values = pickle.load(open("./results/prediction{}_g.pkl".format("_6" if len(actions) == 7 else ""), 'rb'), encoding="bytes")

        elif source =="mov":
            values = pickle.load(open("./results/prediction{}_m.pkl".format("_6" if len(actions) == 7 else ""), 'rb'), encoding="bytes")

        elif source =="ball":
            values = pickle.load(open("./results/prediction_b.pkl", 'rb'), encoding="bytes")

        elif source =="mov_gaze":
            values = pickle.load(open("./results/prediction{}_m_g.pkl".format("_6" if len(actions) == 7 else ""), 'rb'), encoding="bytes")

        elif source =="mov_ball":
            values = pickle.load(open("./results/prediction_m_b.pkl", 'rb'), encoding="bytes")
        
        values = [v for v in values if v[b'label'] in actions]
        corrects_ant = 0.0
        corrects = []
        anticipate = []
        total_frames = []
        for v in values:
            begin,end = v[b'interval']
            probs = v[b'probs']
            label = v[b'label'].numpy()
            pred = v[b'pred'].numpy()
        
            if len(probs.shape)>2:
                vr,h,mi = calc_uncertainties(probs)
                std = probs.std(1)
                probs = probs.mean(1)
                std = std[range(len(c)),c]
            p = np.amax(probs, axis= 1)
            c = np.argmax(probs, axis= 1)
            
            x = np.argmax(p >=0.9)
            #print("{:.1f}/{}- {:.4f} - {:.4f} - {:.4f}".format(vr[x],label,h[x],mi[x],std[x]))
            y = p[x]
            a = c[x]
            if a == label:
                corrects_ant += 1.0

            ant = x if  x > 0 else len(probs)
            ant = ant
            anticipate.append(ant/float(len(probs)))
            total_frames.append(len(probs))
            a = a*c
            corrects.append(pred==label)


            corr = c == label #) * (p>0.9)
            corr_ent = -1*np.sum(probs*np.log(probs),1)
            ratio = len(corr) / 100.0
            for i in range(100):
                p = corr[int(i*ratio)]
                acc_classes[i,s] += float(p)

                p = corr_ent[i] if i < len(corr_ent) else corr_ent[-1]
                entropies[i,s] += float(p)
       
        m = sum(total_frames)/len(total_frames)

        
        print(" {}: {:.2f} - ({:.2f}/{:.2f}% acc)".format(source, (sum(anticipate)/len(anticipate))*m, corrects_ant/len(corrects)*100, sum(corrects)/float(len(corrects))*100) )
    #entropies /= len(actions)*20  
    acc_classes /= len(actions)*20
    observation_ration = np.array([ i / 100.0 for i in range(100)])
    
    axs.plot(observation_ration, acc_classes,)
    axs.axhline(y=th_acc[0], color='B', linestyle='-.')
    axs.axhline(y=th_acc[1], color='k', linestyle='--')
    axs.legend(titles_acc, loc='lower right')
    axs.set_title("Extended dataset - 12 Actions")
    axs.set_xlabel("Observation Ratio")
    axs.set_ylabel("Accuracy")
    
    # d = loadmat("./results/paul.fig",squeeze_me=True, struct_as_record=False)
    # matfig = d['hgS_070000']
    # childs = matfig.children
    # ax1 = [c for c in childs if c.type == 'axes']
    # if(len(ax1) > 0):
    #     ax1 = ax1[0]
    # legs = [c for c in childs if c.type == 'scribe.legend']
    # if(len(legs) > 0):
    #     legs = legs[0]
    # else:
    #     legs=0
   
    
    # # titles_acc = ["3D Pose (Movement)", "3D Pose + Gaze",  "{}% of accuracy".format(int(th_acc[0]*100)), "{}% of accuracy".format(int(th_acc[1]*100))]
    # for line in ax1.children:
    #     if line.type == 'graph2d.lineseries':
    #         #x = line.properties.XData
            
    #         y = line.properties.YData
    #         total = len(y)
    #         x = np.array([ i / float(total) for i in range(total)])
    #         axs[1].plot(x,y)
    # axs[1].axhline(y=th_acc[0], color='B', linestyle='-.')
    # axs[1].axhline(y=th_acc[1], color='k', linestyle='--')
    # axs[1].legend(titles_acc, loc='lower right')
    # axs[1].set_title("Schydlo et al., 2018")
    # axs[1].set_xlabel("Observation Ratio")
    # axs[1].set_ylabel("Accuracy")

    #fig.suptitle("Complete Dataset - All Information", fontsize=15, weight="bold")

    #plt.title("Actions {}".format(",".join([str(i +1) for i in actions])))
    #plt.title("All Actions")
    plt.show()


def entropy(probs):
    return -(probs*np.log(probs)).sum()

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



if __name__ == "__main__": 
    # show_video()
   # generate_video()
    # plot_probability_threshold()
    plot_conf_matrix()
    #plot_uncertainty_threshold()
    #plot_uncertainty()
    #plot_all_charts()
    #plot_acc_classes()
    # plot_uncertainty()

      




