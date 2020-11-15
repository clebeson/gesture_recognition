import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_uncertainty_threshold():
        
    
    values_u = pickle.load(open("results/results_dynamic_star_rgb_hand_MCLSTM_9745.pkl", 'rb'), encoding="bytes")
    results = []
    for t in range(61):
        predictions = {"classes":[], "labels":[]}   
        t /= 10.0
        corrects = []
        total_frames = []
        corrects_ant = 0
        anticipate = []
        for i,value in enumerate(values_u):
            label = value['label'][0]
            pred = value['pred'][0]
            # begin,end = value['interval']
            probs = value['probs']
            probs = np.transpose(value['probs'], (1,0,2))

            vr,h,mi = calc_uncertainties(probs)
            
            meanst = probs.mean(1)
            # print(probs.shape)
            # print(meanst.shape)
            std = probs.std(1)
            c = np.argmax(meanst, axis= 1)

            x = np.argmax(mi<=0.2)
            a = c[x]
            if a == label:
                corrects_ant += 1.0
            if x == 0: 
                # label = 20
                a = 20 #not anticipated
            
        

            predictions["labels"].append(label)
            predictions["classes"].append(a)

            ant = float(x)/len(probs) if  x > 0 else 0 #len(probs)
            anticipate.append(ant)
            total_frames.append(len(probs))

            corrects.append(pred==label)
        plot_conf_matrix(predictions)
        return
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
    
    axis.text(u_acc+0.01, acc_frame+0.01,"{}% de Frames".format(int(acc_frame*100)))
    axis.text(u_frame+0.01,frame-0.04,"{}% de Frames".format(int(frame*100)))


    # axis.set_xlabel("Uncertainty (Mutual Information)")
    # axis.set_ylabel("Anticipation Accuracy / Observation Ratio(OR)")
    # axis.legend([g,o,k,b],["Anticipation Accuracy", "Average OR Anticipation","Maximum Anticipation Accuracy","Minimum Average OR Anticipation"],loc="Upper right",framealpha=1, frameon=True, fancybox=True)
    # axis.set_title("Anticipation vs Uncertainty ($BLSTM_{MC}$)", weight="bold")

    axis.set_xlabel("Incerteza (Informação Mútua)")
    axis.set_ylabel("Acurácia de Antecipação / Taxa Média de Observação (TMO)")
    axis.legend([g,o,k,b],["Acurácia de antecipação", "TMO da antecipação","Acurácia máxima de antecipação","Mínimo TMO de antecipação"],loc="upper right", framealpha=1, frameon=True)
    axis.set_title("Antecipação vs Incerteza (BStar iRGB$_{hand}$)", weight="bold")
    plt.show()

def split_predicition(pred, label, unc, shift,  results ):
    predictions = np.zeros(len(pred))
    # predictions = []
    b = 0
    for i in range(1,len(pred)):
        if pred[i] > 0 and pred[i-1] == 0:
            b = i
        elif pred[i] == 0 and pred[i-1] > 0:    
            if (i-b) > 5:
                x = np.argmax(unc[b:i]<01.4)  
                # print(min(unc[b:i]))    
                results.append([(pred[x+b] == label[x+b]) and x>0,x+b,i-b,pred[x+b] if x>0 and x<(i-b-1) else 0])
                predictions[b-shift:i] = pred[b-shift:i] 
            # print(i-b)
    return np.array(predictions)



if __name__ == "__main__":
    class_names = ["Rótulo binário","Pedido de ajuda","Venha aqui","Pode sair","Siga-me","Pare","Abortar missão","Bom","Não","Ruim","Dar passagem","Apontar","Dúvida","Mais alto","Mais baixo","Silêncio","Antecipação Correto","Antecipação Equivocada"]
    colors = ["green", "black", "blue", "red","brown", "indigo","coral","lime","orange","yellow", "navy","salmon","darkorange","darkblue","darkgreen"]
    det = np.load("./RTGR/complete_results.npy")
    file = "results_splited_8371.pkl"
    data = pickle.load(open(file,"rb"))
    
    corrects = []
    JI_i = 0
    JI_u  = 0
    corrects = []
    tam = 6
    chart = -1
    fig, axs = plt.subplots(tam)
    axis = axs.reshape(-1)
    lines = [ None for _ in range(18)]
    for n, r in data:
            corrects = []
            pred = np.array(r["pred"])
            label = np.array(r["label"])
            unc = np.array(r["uncertainty"])
            pred = split_predicition(pred, label, unc,0, corrects)
            JI_i += sum(np.logical_and(label>0,pred==label))
            JI_u +=  sum(np.logical_or(label>0,pred>0))
            chart += 1               
            if chart == tam:
                plt.legend(handles = lines, labels = class_names, loc='center', ncol = 6, shadow=True,)
                plt.show()
                fig, axs = plt.subplots(tam)
                axis = axs.reshape(-1)
                chart = 0
            ax = axis[chart]
            
            lines[0], = ax.plot(np.where(label>0,1,0),":",color ="k")
            ax.set_title(class_names[int(label[np.argmax(label>0)])])
            
            for g in range(1,16):
                p = np.where(pred==g,1,0)
                if p.sum() == 0:
                    lines[g], = ax.plot(0,0, color = colors[g-1])
                    continue
                b = 0
                for i in range(1,len(pred)):

                    if p[i]== 1 and  p[i-1] == 0:
                        b = i
                    elif p[i] ==0 and p[i-1] == 1: 
                        # if b-1 >0:
                        #     b -=1
                        # if i + 1 <len(p):
                        #     i += 1 
                        # print(b,i)
                        lines[g], = ax.plot(range(b,i),p[b:i], linewidth=2,  color = colors[g-1])

            for cor in corrects:
                if cor[-1]>0:
                    if cor[0]:
                        lines[16], = ax.plot([cor[1],cor[1]], [0,1], "--",color ="g")
                    else:
                        lines[17], = ax.plot([cor[1],cor[1]], [0,1], "--", color ="r")
    
    # plt.show()
            
                # if p>0 and label[i+1]==0:
                #     corrects.append(p==l)
    corrects = np.array(corrects)
    print("Acc Ant =", sum(corrects[:,0])/len(corrects[:,0]),len(corrects),sum(corrects[:,1])/sum(corrects[:,2]))
    print("JI = ",JI_i/JI_u)
    

    
    pred = det[:,0]
    label = det[:,2]
    corrects = []
    # pred = split_predicition(pred, label, det[:,-1], 0, corrects)
    JI_i = sum(np.logical_and(label>0,pred==label))
    JI_u =  sum(np.logical_or(label>0,pred>0))
    corrects = np.array(corrects)
    # print("Acc Ant =", sum(corrects[:,0])/len(corrects[:,0]),len(corrects),sum(corrects[:,1])/sum(corrects[:,2]))
    print("JI = ",JI_i/JI_u)
    print("Acc spt =",sum((pred>0)==(label>0))/len(pred),len(pred))
    print("Acc spt =",sum(pred ==label)/len(pred),len(pred))




            # acc = 100*sum(p==l)/len(p)
            # acc_spt = 100*sum((p>0)==(l>0))/len(p)
            # if acc >90:
            #     print("{:.2f} {:.2f} {}".format(acc,acc_spt,n))






