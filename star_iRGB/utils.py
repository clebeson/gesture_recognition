import requests
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import iteractive_starRGB_hand_dataset as starRGB_dataset
from iteractive_starRGB_hand_model import *
import pickle

def __telegram_bot_sendtext(bot_message):
    try:   
        bot_token = '********************'
        bot_chatID = '*******************'
        send_text = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(bot_token,bot_chatID,bot_message)
        response = requests.get(send_text)
        return response.json()
    except:
        return None

def log(message, telegram = True):
    print(message)
    if telegram:__telegram_bot_sendtext(message)


def __get_params(params):
    string = ""
    for key in params.keys():
        if key not in ["model","name"]:
            string += "  {} = {}".format(key,params[key])
    return string

def __one_iter(params):
        start = timer()
        model_tf= params["model"]
        log( "Hyperparameters = {}".format(__get_params(params)))
        model = IterStarRGBHandModel(
                output_size =  params["num_classes"], 
                hidden_dim =  params["hidden_dim"], 
                n_layers =  params["n_layers"],  
                mode =  "DET",
                dropout = params["fc_drop"],
                )
        model.mov = model_tf.mov
        model.hand = model_tf.hand

        loaders = starRGB_dataset.create_datasets(num_workers=30, batch_size=params["batch_size"], alpha = params["alpha"], window = params["window"], 
                                                  max_size =params["max_seq"][1], spotting = model.output_size == 2)
        model.fit(loaders["train"],loaders["val"],params)
        try:
            model,params = model.load_model("{}_val.pth".format(params["name"]))
        except: pass

        test_data = loaders["test"]
        results = []
        acc = model.predict(test_data,results)
        with open('results_{}_{}.pkl'.format(params["name"], int(acc*10000)), 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        log( "Teste acc = {:.3f}".format(acc*100))
        
        dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
        name = "{}_{}.pth".format(params["name"], int(acc*10000))      
        torch.save(dict_save, name)   
        
        del loaders["train"] 
        del loaders["val"] 
        del loaders["test"] 

        end = timer()
        elapsed = end - start
        h = int(elapsed//3600)
        m = int((elapsed - h*3600)//60)
        s = int(elapsed - h*3600 - m*60)

        log("Elapsed time {}:{}:{}".format(str(h).zfill(2),str(m).zfill(2),str(s).zfill(2)))

        return {'loss': 1-acc, 'status': STATUS_OK, 'params': params}



def bayesian_optimization(model,params, iters = 20):
    trials = Trials()
    params["model"] = model
    best = fmin(fn=__one_iter,
                space=params,
                algo=tpe.suggest,
                max_evals=iters,
                trials=trials,
                show_progressbar = True,
                )
    with open('best_BO_{}-{}.pkl'.format(params["name"],int(acc*10000)), 'wb') as file:
        pickle.dump(best, file, protocol=pickle.HIGHEST_PROTOCOL)