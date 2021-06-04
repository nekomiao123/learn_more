import csv
import itertools
import random

model1 = csv.reader(open('result/newnew_ensemble.csv', 'r'), delimiter=',')
model2 = csv.reader(open('result/label_vgg16bn_pred.csv', 'r'), delimiter=',')
model3 = csv.reader(open('result/dual_res34_pred.csv', 'r'), delimiter=',')
model4 = csv.reader(open('result/efficientnet-b4_pred.csv', 'r'), delimiter=',')

ensemble = []

def voting(*args):
    vote = {}
    for it in args:
        if it not in vote:
            vote[it] = 1
        else:
            vote[it] += 1
    m = max(vote,key=vote.get)
    return m

for m1,m2,m3,m4 in zip(model1,model2,model3,model4):
    img1, label1 = m1
    img2, label2 = m1
    img3, label3 = m1
    img4, label4 = m1

    d = voting(label1,label2,label3,label4)
    ensemble.append(img1+","+d)

# print(ensemble)
with open('./result/best_ensemble.csv','w') as f:
    for i in ensemble:
        f.write(i + '\n')

print("done")

