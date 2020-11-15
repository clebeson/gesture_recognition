
import numpy as np 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import random
from models import *
import threading
from skl_dataset import *
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
np.random.seed(30)
random.seed(30)
have_ipython = False


class BayesianOptmizer(object):
    def __init__(self):
        self.act_dataset = ISDataset()
        self.model_name = {
            "BLSTM":  (20,train_LSTM,predict_BLSTM),
            "DLSTM":  (50,train_LSTM,predict_DLSTM),
            "MLP":  (50,MLP_train,MLP_test),
            "HMM":  (15,HMM_train,HMM_test),
            "NSVM": (50,NSVM_train,SVM_test),
            "SVM":  (50,SVM_train,SVM_test),
            "NB":   (10,NB_train,NB_test),
            "CONV": (50,CONV_train,CONV_test)
            }
    def get_hyperparameters(self,model):
        
        if model == "NB":
            return {
                "max_seq": hp.choice('max_seq',[16,24,32]),
                
            }
        elif model =="HMM":
            return {
                "max_seq": hp.choice('max_seq',[16,24,32]), 
                "states":hp.choice('states', [2,3,4,5,6]),
            }
        elif model =="MLP":
            return  {
                "max_seq": hp.choice('max_seq',[16,24,32]), 
                "hidden_dim":hp.choice('hidden_dim',[(16,), (32,), (48,), (64,), (96,), (128,), (256,), (512,)]), 
                "lr":hp.choice('lr', [1e-1,1e-2,1e-3]),
                
            }
        elif model == "RNN":
            return  {
                "max_seq": hp.choice('max_seq',[16,24,32]), 
                "seq_len":hp.choice('seq_len',[8,16,32]), 
                "batch": hp.choice('batch',[4,8,16,32,64]),  
                "rnn_hidden":hp.choice('rnn_hidden',[32,64,128,256]), 
                "mlp_hidden":hp.choice('mlp_hidden',[32,64,128,256]),
                "lr":hp.choice('lr', [5e-1,1e-1,1e-2,1e-3]),
            }
        elif model == "SVM":
            return  {
                "max_seq": hp.choice('max_seq',[16,24,32]), 
                "loss": hp.choice('loss',["hinge", "squared_hinge"]),
                "c": hp.uniform('c', 0.1, 2),
                "multi_class": hp.choice('multi_class',["ovr", "crammer_singer"])
            }
        elif model == "NSVM":
            return  {
                "max_seq": hp.choice('max_seq',[16,24,32]), 
                "degree": hp.choice('degree',[3,4,5,6,7,8]),
                "kernel": hp.choice('kernel',["linear", "poly", "rbf", "sigmoid"])
            }
        elif "LSTM" in model:
            return {
            "num_classes":15,
            "data_type":["m","g","o"],
            "input_size":21,
            "epoch":300,
            "gpu":0,
            "batch_size":hp.choice("batch_size",[1,2,3,4,5,10,20]),
            # "seq":hp.choice("seq",[(32,96),(64,128),(96,96),(128,128),(32,128)]),
            "max_seq": hp.choice("max_seq",[(8,32),(12,24),(32,32),(24,24),(16,32)]),
            "n_layers":hp.choice("n_layers",[1,2]),
            "hidden_dim":hp.choice("hidden_dim",[128,256,512]),
            "rnn_drop":hp.uniform("rnn_drop",0.1,0.2),
            "fc_drop":hp.uniform("fc_drop",0.1,0.3),
            "clip":hp.uniform("clip",0.1,5),
            "lr":hp.choice("lr",[1e-1,5e-2,5e-3]),
            "weight_decay":hp.choice("weight_decay",[1e-4,1e-5]),
            }

        #CONV
        return  {
            "max_seq": hp.choice('max_seq',[16,24,32]),
            "batch": hp.choice('batch',[4,8,16,32,64]), 
            "out_ch1":hp.choice('out_ch1',[8,16,32,64]), 
            "gpu":0,
            "out_ch2":hp.choice('out_ch2',[8,16,32,64]), 
            "out_ch3":hp.choice('out_ch3',[8,16,32,64]), 
            "kernel1":hp.choice('kernel1',[3,5,7,9]), 
            "kernel2":hp.choice('kernel2',[3,5]), 
            "kernel3":hp.choice('kernel3',[3,5]), 
            "hidden_layer":hp.choice('hidden_layer',[16,32,64,128,256]), 
            "lr":hp.choice('lr', [1e-1,1e-2,1e-3,1e-4]),
            "weight_decay":hp.choice('weight_decay', [1e-1,1e-2,1e-3,1e-4])
        }



    def _optimize(self,params):
            # act_dataset = params["data"]
            print(params)
            model = params["model"]
            cross_acc = 0
            k = 10
            results = []
            train_clf, test_clf = self.model_name[params["model"]][1:]
            self.act_dataset.max_seq = params["max_seq"]
            #cross validaation
            for  train_data, val_data in  self.act_dataset.cross_validation(k):
                model = train_clf(train_data, params)
                acc = test_clf(model,val_data,params,results)
                cross_acc += acc
                # print("------ test acc = {:.2f}% --------".format(acc*100))
                if acc<0.9:
                    cross_acc = acc*k
                    break
            cross_acc/=k
            p = {}
            for key in params.keys():
                if key =="data" or key == "model":continue
                p[key] = params[key]

            # print("cross acc {:.2f}".format(cross_acc*100))
            return {'loss': 1-cross_acc, 'status': STATUS_OK, 'params': p}

    def bayesian_optimization(self,params):
        print("Optimizing {} ...".format(params["name"]))
        trials = Trials()
        iters = self.model_name[params["model"]][0]
        
        best = fmin(fn=self._optimize,
                    space=params,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials=trials,
                    show_progressbar = True,
                    )
        # params = pickle.load(open("results/params_{}.pkl".format(params["name"]),"rb"))
        # print(params)
        # params["batch_size_6"] = 99
        # params["batch_size_12"] = 198
        # params["lr"] = 1e-2
        # params["max_seq"] = (100,100)
        # params["weight_decay"] = 1e-5
        # params["clip"] = 3.0
        # params["hidden_dim"] = 128
        # params["rnn_drop"] = 0.1
        # params["fc_drop"] = 0.1
        # params["epoch"] = 1000
        train = self.act_dataset.get_train()
        test = self.act_dataset.get_test()
        # print(len(train),len(test))
        train_clf, test_clf = self.model_name[params["model"]][1:]
        results = []
        params = space_eval(params,best)
        print(params)
        # params["batch_size"] = len()
        final_model = train_clf(train,params )

        acc = test_clf(final_model,test,params,results)
        print("{} test acc {:.2f}".format(params["name"], acc*100))
        # pickle.dump(trials,open("results/trials_{}.pkl".format(params["name"]),"wb"))
        # pickle.dump(params,open("results/params_{}.pkl".format(params["name"]),"wb"))
        pickle.dump({"acc":acc, "results":results},open("results/results_{}.pkl".format(params["name"]),"wb"))



def lstm_experiments(i,gpu):
    opt = BayesianOptmizer()
    model, name,num_classes,data_type = proposed_models[i]
    # opt.act_dataset.decode(list(range(num_classes)),all_features)
    hyperparameters = opt.get_hyperparameters(model)
    hyperparameters["model"] = model
    hyperparameters["gpu"] = gpu
    hyperparameters["name"] = name
    hyperparameters["num_classes"] = num_classes
    hyperparameters["data_type"] = data_type
    opt.bayesian_optimization(hyperparameters)

def baseline_experiments(i):
    opt = BayesianOptmizer()
    model = baseline_models[i]
    # opt.act_dataset.decode(list(range(num_classes)), features)
    hyperparameters = opt.get_hyperparameters(model)
    hyperparameters["num_features"] = 21
    hyperparameters["num_classes"] = 15
    hyperparameters["model"] = model
    hyperparameters["name"] = model
    opt.bayesian_optimization(hyperparameters)



if __name__ == "__main__":
    # data = np.load("./datasets/acticipate/labels_normalized.npy")
    obj = [2,3]
    head = [30,31,32,33,34,35,36,37, 4, 5]
    mov  = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41,42,43]
    all_features = [2,3,30,31,32,33,34,35,36,37, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41,42,43]

    baseline_models = ["MLP","NSVM","SVM","NB","CONV","HMM"]               
    
    proposed_models =  [
                        ("DLSTM","RTGR_DLSTM",15, ["m","g"]),
                        # ("DLSTM","DLSTM_6_m",15, ["m"]),
                        # ("DLSTM","DLSTM_12_m",12, ["m"]),
                        # ("DLSTM","DLSTM_12_g",12, ["g"]),
                        # # ("DLSTM","DLSTM_12_o",12, ["o"]),
                        # ("DLSTM","DLSTM_12_m_g",12, ["m","g"]),
                        # # ("DLSTM","DLSTM_12_m_o",12, ["m","o"]),
                        # # ("DLSTM","DLSTM_12_m_g_o",12, ["m","g","o"]),
                        # ("BLSTM","BLSTM_12_m_g_o",12, ["m","g","o"]),
                        # # ("BLSTM","BLSTM_12_m_g_o",12, ["m","g","o"]),
                        # # ("BLSTM","BLSTM_12_m_g_o",12, ["m","g","o"]),
                       ]
    
  
    # for (model, name,num_classes,data_type) in proposed_models:
    #     opt = BayesianOptmizer()
    #     opt.act_dataset.decode(list(range(num_classes)),all_features)
    #     hyperparameters = opt.get_hyperparameters(model)
    #     hyperparameters["model"] = model
    #     hyperparameters["gpu"] = 0
    
    #     hyperparameters["name"] = name
    #     hyperparameters["num_classes"] = num_classes
    #     hyperparameters["data_type"] = data_type
    #     opt.bayesian_optimization(hyperparameters)



    # print(len(baseline_models))

    # for model, num_classes, name, features in baseline_models:#
    #     act_dataset.decode(list(range(num_classes)), features)
    #     hyperparameters = get_hyperparameters(model)
    #     hyperparameters["data"] = act_dataset
    #     hyperparameters["num_classes"] = num_classes
    #     hyperparameters["num_features"] = len(features)
    #     hyperparameters["model"] = model
    #     hyperparameters["name"] = name
    #     optimize(hyperparameters)


    # GPUS = [0,1,2,3]*3
    # lstm_threads = [threading.Thread(target=lstm_experiments, args=(i,gpu)) for i, gpu in enumerate(GPUS[:-1])]
    # for t in lstm_threads:
    #     t.start()
    for i in range(len(baseline_models)):
        baseline_experiments(i)

    # print(sys.argv)
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == "LSTM":
    #         lstm_experiments(int(sys.argv[2]),int(sys.argv[3]))
    # else:
    #     baseline_threads = [threading.Thread(target=baseline_experiments, args=(i,)) for i in range(len(baseline_models))]

    #     for t in baseline_threads:
    #         t.start()
    # import pickle
    # import glob
    # files = glob.glob("results/results_*LSTM*")
    # print(len(files))
    # for f in files:
    #     p = pickle.load(open(f,"rb"))["acc"]
    #     print("{} = {:.2f}%".format(f,p*100))
    



        

    

