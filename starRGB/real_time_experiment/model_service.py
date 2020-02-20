from concurrent import futures
import logging

import grpc
import torch
import  image_pb2
import image_pb2_grpc 
import numpy as np
from cabrunco_recognition_grit_94_58 import *


def crop_center(img, out = (110,120)):
        y,x = img.shape[0], img.shape[1]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty

        return img[starty:starty+cropy,startx:startx+cropx,:]

class Model(image_pb2_grpc.ModelServicer):
    def __init__(self,model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = _load_checkpoint(model_path).to(self.device)




    def predict(self, msg, context):

        image = np.frombuffer(msg.image_data, "uint8").reshape(msg.height, msg.width, 3)
        image = sk.transform.resize(image, (110,140))
        image = crop_center(image)
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        image = torch.unsqueeze(torch.from_numpy(image),0)/255.0
        image= image.to(self.device)
        output = self.model(image)
        probs = nn.Softmax(1)(output)
        probs =  np.squeeze(probs.detach().cpu().numpy())
        return image_pb2.Prediction(probs=probs.tobytes(), shape = probs.shape)

def _load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    print("Acc: ",checkpoint["acc"])
    models_to_ensamble = [
                    # {"name":"vgg", "model":models.vgg16_bn(pretrained=False)},
                    {"name":"resnet18", "model":models.resnet50(pretrained=False)}, 
                    {"name":"resnet18", "model":models.resnet101(pretrained=False)}, 
                    ]
    model = Ensemble(models_to_ensamble, name="vgg_resnet")
    # model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

    # def predict_test(self, msg, context):
    #     image = np.frombuffer(msg.image_data).reshape(msg.height, msg.width, 3)
    #     print("Receved image size", image.shape)
    #     probs = image_pb2.Prediction(probs=np.zeros((1,20)).tobytes(), shape = (1,20))
    #     return probs


def serve():
    print("Starting service")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_pb2_grpc.add_ModelServicer_to_server(Model("/notebooks/cabrunco/pytorch/gesture_star/ensemble_star.pt"), server)
    server.add_insecure_port('[::]:3030')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()