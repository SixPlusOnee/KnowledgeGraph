import os
import random


def load_fb15k_data(shuffle=False):
    data_path = os.path.join(os.path.dirname(__file__), "FB15k-237")
    e2i = load_entities_or_relations(os.path.join(data_path, "entity2id.txt"))
    r2i = load_entities_or_relations(os.path.join(data_path, "relation2id.txt"))
    train = load_triples(os.path.join(data_path, "train.txt"), e2i, r2i)
    valid = load_triples(os.path.join(data_path, "valid.txt"), e2i, r2i)
    test = load_triples(os.path.join(data_path, "test.txt"), e2i, r2i)

    if shuffle:
        random.shuffle(train)
        random.shuffle(valid)
        random.shuffle(test)

    return e2i, r2i, train, valid, test


def load_entities_or_relations(file_path):
    with open(file_path) as f:
        lines = [line.split() for line in f.readlines()]
    return {name: int(i) for name, i in lines}


def load_triples(file_path, e2i, r2i):
    with open(file_path) as f:
        lines = [line.split() for line in f.readlines()]
    return [(e2i[s], r2i[r], e2i[t]) for s, r, t in lines]


if __name__ == '__main__':
    load_fb15k_data()
