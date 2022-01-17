import joblib
import numpy as np

TRAIN_DATA_FILE = "train_500.txt"
TEST_DATA_FILE = "test_500.txt"


def compute_dist(test,train):
    a = np.array(test[2])
    b = np.array(train[2])
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    # dist = np.linalg.norm(test[2], train[2])
    dist = np.linalg.norm(a-b)
    return (train[0], dist)


def compute_accuracy(test_table, train_table):
    mean_reciproctal = 0
    accuracy = 0
    for test_row in test_table:
        dists = map(lambda x:compute_dist(test_row,x), train_table)
        dists = list(dists)
        dists.sort(key=lambda x:x[1])
        
        # Compute the mean reciproctal rank
        for i in range(len(dists)):
            if dists[i][0] == test_row[0]:
                mean_reciproctal += 1/(i+1)
                break

        # Compute how often in top 3
        dists = dists[:3]
        for cat,dist in dists:
            if cat==test_row[0]:
                accuracy+=1
                break

    mean_reciproctal /= len(test_table)
    accuracy/=len(test_table)
    return (mean_reciproctal, accuracy)

# Load data
train_table = joblib.load(TRAIN_DATA_FILE)
test_table = joblib.load(TEST_DATA_FILE)

# Compute accuracy for test
mean_rec_test, mean_top3_test = compute_accuracy(test_table, train_table)
print(f"mean_reciproctal_test={mean_rec_test}")
print(f"mean_top3_train={mean_top3_test}")

# Compute accuracy for train 
# Note that both should be 1 i.e. 100%
# Because we are using the same set
mean_rec_train, mean_top3_train = compute_accuracy(train_table, train_table)
print(f"mean_reciproctal_train={mean_rec_train}")
print(f"mean_top3_train={mean_top3_train}")
