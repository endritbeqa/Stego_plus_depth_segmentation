import os
from os.path import join

import numpy as np

def readResults(root): #return an ndArray with IoU,precision, recall for every instance of every Image. of shape (Metric, Image, Instance)

    metrics = []

    filenames = ["AP", "AR", "Pixel_IoU"]

    for i,name in enumerate(filenames):

        file = open(root+name+".txt", "r")
        data = file.read()
        data = data.split("\n")
        data.pop(-1)#remove last element since it is empty
        data = [row.split(":")[1] for row in data]
        data = [row.split(",") for row in data]
        [row.pop(-1) for row in data]
        for j, row in enumerate(data):
            for k, col in enumerate(row):
                data[j][k] = float(data[j][k])

        print(name+": "+str())

    return metrics


def read_metrics(dir_path):

    side_texts = []

    ap_filepath = join(dir_path, "Metrics", "AP.txt")
    ar_filepath = join(dir_path, "Metrics", "AR.txt")
    IoU_filepath = join(dir_path, "Metrics", "Pixel_IoU.txt")

    file_ap = open(ap_filepath, "r")
    file_ar = open(ar_filepath, "r")
    file_IoU = open(IoU_filepath, "r")

    files = [file_ap, file_ar, file_IoU]
    metric_type = ["AP", "AR", "Mean IoU"]

    for i, file in enumerate(files):
        data = file.read()
        values = data.split(",")
        values.pop(-1)
        values = [float(val.split(":")[1]) for val in values]
        print(metric_type+str(sum(values)/len(values)))
        side_texts.append(values)

    side_texts = np.array(side_texts).transpose()

    return side_texts


def AUC(metrics_N_M, metrics_1_1):
    IoU1_1 = np.array(metrics_1_1[0])
    IoUN_M = np.array(metrics_N_M[0])

    return IoUN_M







if __name__=="__main__":

    root ="/home/endrit/geo-scene/results/predictions/BGMM_projected_good_images/Metrics/"

    results = readResults(root)
    j=    AUC(results,results)

    print("breakpoint")




