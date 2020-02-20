from __future__ import print_function
import logging

import grpc

import image_pb2
import image_pb2_grpc
import time
import numpy as np


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    options = [('Grpc.Core.ChannelOptions.MaxMessageLength', (640 * 480 * 4 * 8))]
    with grpc.insecure_channel('10.10.2.2:3030', options = options) as channel:
        stub = image_pb2_grpc.ModelStub(channel)
    for i in range(30):
        image = image_pb2.Image(image_data = np.zeros((120,160,3)).tobytes(),width = 160, height = 120, frame = i)
        response = stub.predict(image)
        pred = response
        probs = np.frombuffer(pred.probs).reshape(tuple(pred.shape))
        print("Recieved image size", probs.shape)
        time.sleep(2)


if __name__ == '__main__':
    logging.basicConfig()
    run()
