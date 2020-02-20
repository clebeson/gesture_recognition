import pickle
import matplotlib.pyplot as plt
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fps', type=int, default=10, help='fps')
parser.add_argument('--gesture', type=int, default=1, help='gesture')
parser.add_argument('--pred', type=str, default="all", help='predictions (gesture and Non gestures)')
args = parser.parse_args()

fps = args.fps-1
g = args.gesture-1
pred = list(range(16)) if args.pred == "all" else  list(range(1,16)) if args.pred == "g" else [0]
#print(pred)
r = pickle.load(open("results_complete.pkl","rb"))
values = [u for p,u in r[fps][g] if p in pred]
plt.plot(values)
plt.show()
