import joblib
import numpy as np

train_table = joblib.load("train.txt")
test_table = joblib.load("test.txt")

def compute_dist(test,train):
    a = np.array(test[2])
    b = np.array(train[2])
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    # dist = np.linalg.norm(test[2], train[2])
    dist = np.linalg.norm(a-b)
    return (train[0], dist)


accuracy = 0
for test_row in test_table:
    dists = map(lambda x:compute_dist(test_row,x), train_table)
    dists = list(dists)
    dists.sort(key=lambda x:x[1])
    dists = dists[:3]
    for cat,dist in dists:
        if cat==test_row[0]:
            accuracy+=1
            break

accuracy/=len(test_table)
print(f"accuracy={accuracy}")
    