from generate_star_complete import * 
import cv2
import grpc
import  image_pb2
from  image_pb2_grpc import ModelStub
import time
import copy
white = np.uint8(np.ones((480,640,3))*255)

cap = cv2.VideoCapture(0)
classes = ["abort","circle","hello","no","stop","turn","turn left","turn right","warn"]
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (10, 50)  
fontScale = 0.7
color = (255, 255, 255) 
thickness = 1

options = [('Grpc.Core.ChannelOptions.MaxMessageLength', (640 * 480 * 4 * 8))]
channel = grpc.insecure_channel('10.10.2.2:3030', options = options)
model = ModelStub(channel)

ret, frame = cap.read()
before = frame
before = cv2.blur(before,(5,5))
before = m.imresize(before,(120,160)).astype(np.float32)
acc = []
white_text = white
while(ret):
    
    cv2.imshow('frame',m.imresize(np.hstack((frame,white_text)),(600,1600)))
    time.sleep(33/1000.)
    ret, frame = cap.read()
    now = cv2.blur(frame,(5,5))
    now = m.imresize(now,(120,160)).astype(np.float32)
    result = cosine_distance(before,now)
    
    before = now
    mean = result.mean()
    shape = ()
        
    if mean > 2:
        white_text = cv2.putText(copy.deepcopy(white), "Buffering...", (150, 300)  , font, 2, (0,0,0), thickness, cv2.LINE_AA)
        #print(mean, "acc")
        acc.append(result)
    elif len(acc) > 20:
        time_star = time.time()
        rest = len(acc)%3
        step = len(acc)//3 
        r = reduce(add, acc[:step] )
        g = reduce(add, acc[step:2*step+rest] )
        b = reduce(add, acc[2*step+rest:]) 
        star =  np.uint8(normalize(np.stack([r,g,b],2)) * 255)
        #star = get_starRGB(acc))*255)
      
        star = m.imresize(star,(120,160))
        time_star  = time.time()-time_star
        time_predict = time.time()
        image = image_pb2.Image(image_data = star.tobytes(), width = 160, height = 120, frame = 1)
        pred = model.predict(image)
        time_predict = time.time()-time_predict
        probs = np.frombuffer(pred.probs,"float32").reshape(tuple(pred.shape))
        p,c = probs.max(),probs.argmax()
        #print("elapsed = {}   prob = {}  class = {}".format(time.time()-start,p,c))
        image = cv2.putText(m.imresize(star,(480,640)), \
        "Elapsed star + pred = {:.3f} + {:.3f} = {:.3f} (s)".format(time_star, time_predict,time_star + time_predict),\
         org, font, fontScale, color, thickness, cv2.LINE_AA)
        image =cv2.putText(m.imresize(image,(480,640)), \
        "prob = {:.2f} class = {}".format(p,classes[c]),\
         (10,80), font, fontScale, color, thickness, cv2.LINE_AA) 
        cv2.imshow('frame',m.imresize(np.hstack([frame,image]),(600,1600)))
        cv2.waitKey(30)
        for _ in range(120):
            time.sleep(0.03)
        del acc[:]
        
    else:
        #print(mean)
        if len(acc) <3 : white_text = white
        else:
            white_text = cv2.putText(copy.deepcopy(white), "No Gesture!", (150,300), font, 2, (0,0,0), thickness, cv2.LINE_AA) 
        # result = np.stack([result,result,result],2)
        # result = m.imresize(result,(480,640))
        # print(frame.shape,result.shape)
        # cv2.imshow('frame',np.hstack((frame,white)))
  
    if cv2.waitKey(1) & 0xFF == ord('q'):break
    

cap.release()
cv2.destroyAllWindows()
