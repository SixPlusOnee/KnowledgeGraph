import os
import random
from datas.data_utils import load_fb15k_data
from TransE.transE import TransE
import tensorflow as tf

if __name__ == '__main__':
    # load data
    e2i, r2i, train, valid, test = load_fb15k_data(shuffle=True)

    # 将train, valid的三元组以dict key形式存储，提高负采样的效率
    train = {(s, r, t): 0 for s, r, t in train}
    valid = {(s, r, t): 0 for s, r, t in valid}

    transE = TransE(entity_nums=len(e2i), relation_nums=len(r2i), embedding_dim=50)

    epochs = 3
    train_batch_size = 400
    valid_batch_size = 100

    checkpoint_dir = './train_ckpt'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(entity_embedding=transE.entity_embedding,
                                     relation_embedding=transE.relation_embedding)

    # train
    for epoch in range(epochs):
        # 每个epoch遍历一次训练集三元组
        batch_num = len(train) // train_batch_size + 1
        train_triples = list(train.keys())
        valid_triples = list(valid.keys())

        batch_start_index = 0
        # 每一步，取train_batch_size个训练集三元组进行训练，再取valid_batch_size个验证集三元组计算valid loss
        for i in range(batch_num):
            batch_end_index = batch_start_index + train_batch_size - 1
            batch_end_index = batch_end_index if batch_end_index < len(train_triples) else len(train_triples) - 1

            train_samples = train_triples[batch_start_index: batch_end_index + 1]
            valid_samples = random.sample(valid_triples, valid_batch_size)

            train_batch = transE.generate_batch(train_samples, train)
            valid_batch = transE.generate_batch(valid_samples, valid)

            train_loss, valid_loss = transE.train_step_per_batch(train_batch, valid_batch)
            print("Batch: {} / {}; Train Loss: {}; Valid Loss {}".format(i + 1,
                                                                         batch_num,
                                                                         round(train_loss, 3),
                                                                         round(valid_loss, 3)))
            batch_start_index += train_batch_size
            if (i + 1) % 100 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

    # test
    transE.test(test)
