import csv
import itertools
import random

model1 = csv.reader(open('result/AdamW_seresnext50_32x4d_pred.csv', 'r'), delimiter=',')
model2 = csv.reader(open('result/AdamW_efficientnet_b3a_pred.csv', 'r'), delimiter=',')
model3 = csv.reader(open('result/AdamW_tf_efficientnetv2_s_in21ft1k_pred.csv', 'r'), delimiter=',')

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

for m1,m2,m3 in zip(model1,model2,model3):
    if m1 == m2:
        ensemble.append(m1)
    else:
        if m2 == m3:
            ensemble.append(m2)
        else:
            ensemble.append(m1)

# print(ensemble)
with open('./result/best_ensemble.csv','w') as f:
    for i in ensemble:
        f.write(str(i[0]) + "," + str(i[1]) + '\n')

print("done")

