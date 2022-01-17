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
    return (train[1], train[0], dist)


accuracy = 0
l = [test_table[45], test_table[1000], test_table[0]]
for test_row in l:
    dists = map(lambda x:compute_dist(test_row,x), train_table)
    dists = list(dists)
    dists.sort(key=lambda x:x[2])
    dists = dists[:3]
    print(f"Filename: {test_row[1]}, Category: {test_row[0]}")
    for d in dists:
        print(f"    Filename: {d[0]}, Category: {d[1]}, Distance: {d[2]}")


