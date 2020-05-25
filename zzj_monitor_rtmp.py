import os
import os.path as osp
import time
import multiprocessing as mp

from process_rtmp import *
max_q_size = 1
queues = [mp.Queue(maxsize=2) for i in range(0, max_q_size+1) ]

processes = []
processes.append(mp.Process(target=load_im, args=(queues[0],None)))
processes.append(mp.Process(target=first_layer_detection, args=(queues[0], queues[1] )))
#processes.append(mp.Process(target=post_process, args=(queues[1], None )))


for process in processes:
    process.daemon = True
    process.start()
for process in processes:
    process.join()

