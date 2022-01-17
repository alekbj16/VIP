import joblib

train_table = joblib.load("train_200.txt")
test_table = joblib.load("test_200.txt")

train = []
for t in train_table:
    train.append(t[0])

train = set(train)

test = []
for t in test_table:
    test.append(t[0])
test = set(test)

#  print("The categories in train")
for cat in train:
    print(f"{cat}")

#  print("The categories in test")
#  for cat in test:
    #  print(f"    {cat}")
