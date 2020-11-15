
import numpy as np 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import random
from models import *
import threading
from timeit import default_timer as timer
from tqdm import tqdm
# from skl_dataset import *
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
np.random.seed(30)
random.seed(30)
have_ipython = False


class BayesianOptmizer(object):
    def __init__(self):
        # self.act_dataset = ISDataset()
        self.model_name = {
            "BLSTM":  (20,train_LSTM,predict_BLSTM),
            "DLSTM":  (50,train_LSTM,predict_DLSTM),
            "MLP":  (50,MLP_train,MLP_test),
            "HMM":  (15,HMM_train,HMM_test),
            "NSVM": (50,NSVM_train,SVM_test),
            "SVM":  (50,SVM_train,SVM_test),
            "NB":   (10,NB_train,NB_test),
            "CONV": (50,CONV_train,CONV_test),
            "DSRGB":(10,train_LSTM,predict_DLSTM)
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
            "epoch":1,
            "gpu":0,
            "batch_size":hp.choice("batch_size",[1,2,3,4,5,10,20]),
            "max_seq": hp.choice("max_seq",[(8,32),(12,24),(32,32),(24,24),(16,32)]),
            "n_layers":hp.choice("n_layers",[1,2]),
            "hidden_dim":hp.choice("hidden_dim",[128,256,512]),
            "rnn_drop":hp.uniform("rnn_drop",0.1,0.2),
            "fc_drop":hp.uniform("fc_drop",0.1,0.3),
            "clip":hp.uniform("clip",0.1,5),
            "lr":hp.choice("lr",[1e-1,5e-2,5e-3]),
            "weight_decay":hp.choice("weight_decay",[1e-4,1e-5]),
            }
        elif "DSRGB" in model: #hands
            return {
            "num_classes":20,
            "gpu":0,
            "epoch":100,
            "batch_size":96,
            "max_seq":(16,32),
            "n_layers":hp.choice("n_layers", [1,2]),
            "hidden_dim":hp.choice("hidden_dim",[512,1024]),
            "fc_drop":0.15,
            "alpha":0.6,
            "window":5,
            "clip":5.0,
            "lr":hp.choice("lr",[[1e-4, 1e-3],[1e-4, 5e-3]]),
            "weight_decay":1e-5,
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



    def DSRGB_optimize(self, params):
            start = timer()
            myprint( "Hyperparameters = {}".format(params))
            # myprint("New optimizing ({})".format(params["name"]))
            loaders = create_datasets(num_workers=24, batch_size=params["batch_size"], alpha = params["alpha"], window = params["window"], max_size =params["max_seq"][1])
            train_clf, test_clf = self.model_name[params["model"]][1:]
            # model,params = load_model("dynamic_star_rgb_hand_9728.pth")
            model = DStarRGBModel(
                output_size =  params["num_classes"], 
                hidden_dim =  params["hidden_dim"], 
                n_layers =  params["n_layers"],  
                mode =  "DET",
                dropout = params["fc_drop"],
                )

            model = train_clf(model,loaders["train"],loaders["val"],params)
            try:
                model,params = load_model("dynamic_star_rgb_RNN_val.pth")
            except:
                pass
            # test_data = DStarRGBHandDataset(dataset="test", max_size = None, alpha = params["alpha"], window = params["window"], 
                            # transform=DataTrasformation(output_size=(110,120), data_aug = False))
            test_data = loaders["test"]
            acc = test_clf(model,test_data,params,[])

            myprint( "Teste acc = {:.3f}".format(acc*100))
            
            dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
            name = "dynamic_star_rgb_RNN_{}.pth".format(int(acc*10000))      
            torch.save(dict_save, name)   
            
            del loaders["train"] 
            del loaders["val"] 
            del loaders["test"] 

            end = timer()
            elapsed = end - start
            h = int(elapsed//3600)
            m = int((elapsed - h*3600)//60)
            s = int(elapsed - h*3600 - m*60)

            myprint("Elapsed time {}:{}:{}".format(str(h).zfill(2),str(m).zfill(2),str(s).zfill(2)))

            return {'loss': 1-acc, 'status': STATUS_OK, 'params': params}


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
       
        trials = Trials()
        iters = self.model_name[params["model"]][0]
        
        best = fmin(fn=self.DSRGB_optimize,
                    space=params,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials=trials,
                    show_progressbar = True,
                    )

        # self.DSRGB_optimize(params)



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
    # dynamic_star_rgb_hand_9728.pth
    opt = BayesianOptmizer()
    model = "DSRGB"
    hyperparameters = opt.get_hyperparameters(model)
    hyperparameters["model"] = model
    hyperparameters["name"] = model
    opt.bayesian_optimization(hyperparameters)
    # opt.DSRGB_optimize(hyperparameters)
    # for lr,nl in [[[1e-4, 1e-3],512],[[1e-4, 1e-3],1024],[[1e-4, 5e-3],512], [[1e-4, 5e-3],512]]:
    #     hyperparameters =  {
    #         "model":model,
    #         "name":model,
    #         "num_classes":20,
    #         "gpu":0,
    #         "epoch":70,
    #         "batch_size":96,
    #         "max_seq":(12,24),
    #         "n_layers":nl,
    #         "hidden_dim":2,
    #         "fc_drop":0.2,
    #         "alpha":0.6,
    #         "window":5,
    #         "clip":5.0,
    #         "lr":lr,
    #         "weight_decay":1e-5,
    #         }
    #     # opt.DSRGB_optimize(hyperparameters)
    #     opt.bayesian_optimization(hyperparameters)


    



        

    

