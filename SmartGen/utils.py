import numpy as np
import random
from collections import defaultdict


def build_index(dataset):
    # Not used later, but required by main
    return {}, {}


def data_partition(dataset):
    """
    Reads data/{dataset}.txt
    Format: user_id item_id
    """
    user_train = defaultdict(list)
    user_valid = {}
    user_test = {}

    path = f"data/{dataset}.txt"
    with open(path, "r") as f:
        for line in f:
            u, i = line.strip().split()
            u = int(u)
            i = int(i)
            user_train[u].append(i)

    # split last item for test, second last for valid
    for u in list(user_train.keys()):
        if len(user_train[u]) < 3:
            user_valid[u] = []
            user_test[u] = []
            continue
        user_valid[u] = [user_train[u].pop()]
        user_test[u] = [user_train[u].pop()]

    usernum = max(user_train.keys()) + 1
    itemnum = max(i for items in user_train.values() for i in items) + 1

    return user_train, user_valid, user_test, usernum, itemnum


class WarpSampler:
    def __init__(self, user_train, usernum, itemnum, batch_size, maxlen, n_workers=1):
        self.user_train = user_train
        self.users = list(user_train.keys())
        self.batch_size = batch_size
        self.itemnum = itemnum
        self.maxlen = maxlen

    def next_batch(self):
        u, seq, pos, neg = [], [], [], []

        for _ in range(self.batch_size):
            user = random.choice(self.users)
            items = self.user_train[user]

            if len(items) < 2:
                continue

            s = items[-self.maxlen:]
            u.append(user)
            seq.append(s[:-1])
            pos.append(s[1:])
            neg.append([random.randint(1, self.itemnum - 1) for _ in s[1:]])

        return u, seq, pos, neg

    def close(self):
        pass


def evaluate(model, dataset, args):
    return 0.0, 0.0


def evaluate_valid(model, dataset, args):
    return 0.0, 0.0

