import random
import argparse
import numpy as np

from preprocessing import parse_annotation,parse_GTSDB_dataset
import json


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.2f,%0.2f, ' % (anchors[i,0], anchors[i,1])

    #there should not be comma after last anchor, that's why
    r += '%0.2f,%0.2f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"

    print r

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print "iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances)))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def main():
    #LABELS = ['traffic_sign']
    LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair","cow","diningtable","dog","horse","motorbike","pottedplant","sheep","sofa","train","tvmonitor","person"]
    #train_imgs_GTSDB,train_labels_GTSDB = parse_GTSDB_dataset()
    # train_imgs, train_labels = parse_annotation('/data4/sunsiyuan/data/LISA/Annotations/',
    #                                             '/data4/sunsiyuan/data/LISA/JPEGImages/',
    #                                              LABELS)
    train_imgs, train_labels = parse_annotation('/data4/sunsiyuan/data/VOC/VOCdevkit/VOC2012/Annotations/',
                                                '/data4/sunsiyuan/data/VOC/VOCdevkit/VOC2012/JPEGImages/',
                                                 LABELS)
    print len(train_imgs)
    print train_labels
    #train_imgs.extend(train_imgs_GTSDB)
    #train_labels['traffic_sign'] =  train_labels['traffic_sign'] + train_labels_GTSDB['traffic_sign']
    num_anchors = 5
    grid_w = 13
    grid_h = 13

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        cell_w = image['width']/grid_w
        cell_h = image['height']/grid_h

        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/cell_w
            relatice_h = (float(obj["ymax"]) - float(obj['ymin']))/cell_h
            #print (relatice_h,relative_w)
            annotation_dims.append(map(float, (relative_w,relatice_h)))
    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print '\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids)
    print_anchors(centroids)

if __name__ == '__main__':
    main()
