from flask import Flask, request
from flask import render_template
import time
import json
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import copy

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240,
               170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50,
               120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)

    sample_points_X.append(points_X[0])
    sample_points_Y.append(points_Y[0])
    l = len(points_X)
    # print(l)
    step = int(l / 50)
    # print(step)
    if step != 0:
        endX = points_X[-1]
        endY = points_Y[-1]
        points_X = points_X[0:-1:step]
        points_Y = points_Y[0:-1:step]
        points_X.append(endX)
        points_Y.append(endY)
    klist = []
    path = 0
    for i in range(0, len(points_X) - 1):
        try:
            klist.append((points_Y[i + 1] - points_Y[i]) / (points_X[i + 1] - points_X[i]))
        except:
            klist.append(0)
        path += math.sqrt(pow(points_X[i + 1] - points_X[i], 2) + pow(points_Y[i + 1] - points_Y[i], 2))
    d = path / 99
    q = 0
    k = klist[q]
    sample_points_X = [points_X[0]]
    sample_points_Y = [points_Y[0]]
    sx = np.sign(points_X[q + 1] - points_X[q])
    sy = np.sign(points_Y[q + 1] - points_Y[q])

    for i in range(99):
        r = math.sqrt(pow(sample_points_X[i] - points_X[q + 1], 2) + pow(sample_points_Y[i] - points_Y[q + 1], 2))
        # print('r=',r,', d=',d)
        if ((r < d) or (r == d)) and ((r != 0) or (d != 0)):
            r = d - r
            q += 1
            try:
                k = klist[q]
                if sx != 0:
                    # print('i=',i,',sx!=0,k=',k)
                    sx = np.sign(points_X[q + 1] - points_X[q])
                    sy = np.sign(points_Y[q + 1] - points_Y[q])
                    dx = math.sqrt(1 / (k * k + 1)) * r
                    dy = math.sqrt(1 / (k * k + 1)) * r * abs(k)
                    sample_points_X.append(points_X[q] + dx * sx)
                    sample_points_Y.append(points_Y[q] + dy * sy)
                elif (sx == 0) and (sy == 0):
                    sx = np.sign(points_X[q + 1] - points_X[q])
                    sy = np.sign(points_Y[q + 1] - points_Y[q])
                    sample_points_X.append(points_X[q])
                    sample_points_Y.append(points_Y[q])
                else:
                    sx = np.sign(points_X[q + 1] - points_X[q])
                    sy = np.sign(points_Y[q + 1] - points_Y[q])
                    sample_points_X.append(points_X[q])
                    sample_points_Y.append(points_Y[q] + r * sy)
                continue
            except:
                sample_points_X.append(points_X[-1])
                sample_points_Y.append(points_Y[-1])
                # print(i)
                break
        elif r == 0 and d == 0:
            for k in range(i, 99):
                sample_points_X.append(points_X[q])
                sample_points_Y.append(points_Y[q])
            break

        if sx != 0:
            dx = math.sqrt(1 / (k * k + 1)) * d
            dy = math.sqrt(1 / (k * k + 1)) * d * abs(k)
            sample_points_X.append(sample_points_X[i] + dx * sx)
            sample_points_Y.append(sample_points_Y[i] + dy * sy)
        else:
            sample_points_X.append(sample_points_X[i])
            sample_points_Y.append(sample_points_Y[i] + d * sy)

    sample_points_X[-2] = points_X[-1]
    sample_points_Y[-2] = points_Y[-1]
    sample_points_X[-1] = points_X[-1]
    sample_points_Y[-1] = points_Y[-1]

    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def dis(x1, x2, y1, y2):
    d = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
    return d


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    norm_v_s_X, norm_v_s_Y = [], []

    # TODO: Set your own pruning threshold
    threshold = 0.8
    # TODO: Do pruning (12 points)

    # norm_gesture
    gest_X = gesture_points_X
    gest_Y = gesture_points_Y
    s = 0
    L = 10
    max_w = max(gest_X)
    min_w = min(gest_X)
    max_h = max(gest_Y)
    min_h = min(gest_Y)

    mean_points_X = (max_w + min_w) / 2
    mean_points_Y = (max_h + min_h) / 2

    s = L / max(max_w, max_h)
    norm_gest_X = []
    norm_gest_Y = []
    for i in gest_X:
        norm_gest_X.append(s * (i - mean_points_X))
    for i in gest_Y:
        norm_gest_Y.append(s * (i - mean_points_Y))

    for i in range(10000):
        # 先norm tpl 的首尾2个
        s = 0
        max_w = max(template_sample_points_X[i])
        min_w = min(template_sample_points_X[i])
        max_h = max(template_sample_points_Y[i])
        min_h = min(template_sample_points_Y[i])

        mean_points_X = (max_w + min_w) / 2
        mean_points_Y = (max_h + min_h) / 2

        s = L / max(max_w, max_h)

        start_X = s * (template_sample_points_X[i][0] - mean_points_X)
        start_Y = s * (template_sample_points_Y[i][0] - mean_points_Y)
        end_X = s * (template_sample_points_X[i][-1] - mean_points_X)
        end_Y = s * (template_sample_points_Y[i][-1] - mean_points_Y)

        if (dis(start_X, norm_gest_X[0], start_Y, norm_gest_Y[0]) or dis(end_X, norm_gest_X[-1], end_Y,
                                                                         norm_gest_Y[-1])) < threshold:
            valid_words.append(words[i])

            valid_template_sample_points_X.append(template_sample_points_X[i])
            valid_template_sample_points_Y.append(template_sample_points_Y[i])

            # norm tpl
            norm_tpl_X = []
            norm_tpl_Y = []

            for j in template_sample_points_X[i]:
                norm_tpl_X.append(s * (j - mean_points_X))

            for j in template_sample_points_Y[i]:
                norm_tpl_Y.append(s * (j - mean_points_Y))

            norm_v_s_X.append(norm_tpl_X)
            norm_v_s_Y.append(norm_tpl_Y)

    # print('validwords长度：',len(valid_words))
    # print('valid_words = ', valid_words)

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, norm_v_s_X, norm_v_s_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                     valid_template_sample_points_Y, valid_words):
    shape_scores = []
    # TODO: Set your own L

    # TODO: Calculate shape scores (12 points)
    # norm_gesture
    gest_X = gesture_sample_points_X
    gest_Y = gesture_sample_points_Y
    s = 0
    L = 10
    max_w = max(gest_X)
    min_w = min(gest_X)
    max_h = max(gest_Y)
    min_h = min(gest_Y)

    mean_points_X = (max_w + min_w) / 2
    mean_points_Y = (max_h + min_h) / 2

    s = L / max(max_w, max_h)
    norm_gest_X = []
    norm_gest_Y = []
    for i in gest_X:
        norm_gest_X.append(s * (i - mean_points_X))
    for i in gest_Y:
        norm_gest_Y.append(s * (i - mean_points_Y))

    valid_tpl_sample_X = valid_template_sample_points_X
    valid_tpl_sample_Y = valid_template_sample_points_Y
    N = 0
    N = len(valid_tpl_sample_X)
    # print('N= ', N)
    for i in range(N):
        Sum = 0
        for j in range(99):
            Sum += dis(norm_gest_X[j], valid_tpl_sample_X[i][j], norm_gest_Y[j], valid_tpl_sample_Y[i][j]) / 100
        shape_scores.append(Sum)

    # print('len of shape_scores:', len(shape_scores))
    # print('min_shape:',min(shape_scores),'max_shape:',max(shape_scores))

    n = 50
    topN_SS = []
    topN_word = []
    re1 = []
    inst_list = copy.deepcopy(shape_scores)
    for i in range(n):  # return n smallest elements
        number = min(inst_list)
        index = inst_list.index(number)
        inst_list[index] = 100
        re1.append(index)
    print(re1)
    '''
    print(heapq.nsmallest(n,shape_scores))
    for i in list(re1):
        topN_word.append(valid_words[i])
    print('shape_topNword: ',topN_word)

    count=1
    for word in topN_word:
        print('x',count,' = ', valid_template_sample_points_X[valid_words.index(word)])
        print('y',count,' = ' , valid_template_sample_points_Y[valid_words.index(word)])
        count+=1
    print('x',count,' = ' ,norm_gest_X)
    print('y',count,' = ' , norm_gest_Y)
    '''

    return shape_scores, re1


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                        valid_template_sample_points_Y, valid_words, re1):
    radius = 0.1
    # TODO: Calculate location scores (12 points)
    gest_X = gesture_sample_points_X
    gest_Y = gesture_sample_points_Y
    valid_tpl_X = valid_template_sample_points_X
    valid_tpl_Y = valid_template_sample_points_Y

    location_scores = []
    a = [0.005] * 100
    a[0] = 0.16
    a[-1] = 0.35

    L = len(re1)

    for k in re1:
        Dpq = 0
        X = 0
        for i in range(99):
            dpq = radius + 1
            dist1 = dis(gest_X[i], valid_tpl_X[k][i], gest_Y[i], valid_tpl_Y[k][i])
            for j in range(99):
                dist = dis(gest_X[i], valid_tpl_X[k][j], gest_Y[i], valid_tpl_Y[k][j])
                dpq = min(dist, dpq)
            X = X + a[i] * dist1
        Dpq = Dpq + max((dpq - radius), 0)
        if (Dpq == 0):
            location_scores.append(0)
        else:
            location_scores.append(X)
    '''
    print(len(location_scores))
    print(sorted(location_scores))
    print('min_loc:', min(location_scores), 'max_loc:', max(location_scores))

    n = 10
    topN_word = []
    re1 = map(location_scores.index, heapq.nsmallest(n, location_scores))
    re2 = map(intergration_scores.index, heapq.nsmallest(n, intergration_scores))
    #print(heapq.nsmallest(n, location_scores))
    for i in list(re1):
        topN_word.append(valid_words[i])
    print('location_topNword: ', topN_word)
    '''

    return location_scores


def get_integration_scores(shape_scores, location_scores, re1):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.9
    # TODO: Set your own location weight
    location_coef = 0.1
    for i in range(len(re1)):
        integration_scores.append(shape_coef * shape_scores[re1[i]] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores, words, re1):
    best_word = 'the'
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    topN_IS = []
    topN_word = []
    # topN_FS = []
    inst_list = copy.deepcopy(integration_scores)
    for i in range(n):  # return n smallest elements
        number = min(inst_list)
        index = inst_list.index(number)
        topN_IS.append(inst_list[index])
        inst_list[index] = 100
        topN_word.append(valid_words[re1[index]])
    print('topN_word: ', topN_word)
    # print('topN_FS:', topN_FS)
    # print('FS_min: ',min(topN_FS))

    if topN_IS[0] == topN_IS[1]:
        best_word = topN_word[0] + ', ' + topN_word[1]
    else:
        best_word = topN_word[0]

    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():
    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    # gesture_points_X = [gesture_points_X]
    # gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    # print('gesture points长度: ',len(gesture_points_X))

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, norm_v_s_X, norm_v_s_Y = do_pruning(
        gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores, re1 = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, norm_v_s_X, norm_v_s_Y,
                                         valid_words)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y,
                                          valid_template_sample_points_X, valid_template_sample_points_Y, valid_words,
                                          re1)

    integration_scores = get_integration_scores(shape_scores, location_scores, re1)

    best_word = get_best_word(valid_words, integration_scores, words, re1)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()