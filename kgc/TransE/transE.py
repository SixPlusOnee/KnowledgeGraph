# -*- coding: utf-8 -*-
"""
transE的tensorflow2.0(keras)实现
"""
import tensorflow as tf
import numpy as np
import random


class TransE:
    def __init__(self, entity_nums, relation_nums, embedding_dim, lr=0.001, margin=1, distance_type="l2"):
        """
        :param entity_nums: 实体数量（实体id范围为 [0, entity_nums-1]
        :param relation_nums: 关系类型数量（关系类型id范围为 [0, relation_nums-1]
        :param embedding_dim: 嵌入维度
        :param lr: 学习率
        :param margin: transE计算loss时的margin
        :param distance_type: "l1" or "l2"
        """

        self.entity_nums = entity_nums
        self.relation_nums = relation_nums
        self.entity_embedding = tf.keras.layers.Embedding(input_dim=self.entity_nums,
                                                          output_dim=embedding_dim)
        self.relation_embedding = tf.keras.layers.Embedding(input_dim=self.relation_nums,
                                                            output_dim=embedding_dim)
        self.margin = margin
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        if distance_type == "l2":
            self.distance = self.distance_l2
        else:
            self.distance = self.distance_l1

    def triple2embeddings(self, s, r, t):
        return self.entity_embedding(s), self.relation_embedding(r), self.entity_embedding(t)

    def distance_l1(self, s, r, t):
        # 计算l1距离
        s, r, t = self.triple2embeddings(s, r, t)
        return tf.reduce_sum(tf.abs(s + r - t))

    def distance_l2(self, s, r, t):
        # 计算l2距离
        s, r, t = self.triple2embeddings(s, r, t)
        return tf.reduce_sum(tf.square(s + r - t))

    def loss_function(self, dist_positive, dist_negative):
        # 计算loss
        return max(0, dist_positive - dist_negative + self.margin)

    def l2_normalize_entity_embeddings(self):
        # 对entity embeddings做l2归一化
        weights = self.entity_embedding.get_weights()[0]
        for i in range(weights.shape[0]):
            weights[i] /= np.linalg.norm(weights[i])

        self.entity_embedding.set_weights([weights])

    def train_step_per_batch(self, train_batch, valid_batch=None):
        # 每个batch的训练步
        train_loss = 0
        with tf.GradientTape() as tape:
            # 对所有样本累加loss
            for triple_positive, triple_negative in train_batch:
                dist_positive = self.distance(*triple_positive)
                dist_negative = self.distance(*triple_negative)
                train_loss += self.loss_function(dist_positive, dist_negative)

        if train_loss > 0:
            # SGD
            variables = self.entity_embedding.trainable_variables + self.relation_embedding.trainable_variables
            gradients = tape.gradient(train_loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            # l2 normalize
            self.l2_normalize_entity_embeddings()

        # 计算valid loss
        valid_loss = 0
        if valid_batch is not None:
            for triple_positive, triple_negative in valid_batch:
                dist_positive = self.distance(*triple_positive)
                dist_negative = self.distance(*triple_negative)
                valid_loss += self.loss_function(dist_positive, dist_negative)

        train_loss /= len(train_batch)
        valid_loss /= len(valid_batch)
        return float(train_loss), float(valid_loss)

    def generate_batch(self, triples, triples_pool):
        batch = dict()
        for positive_triple in triples:
            negative_triple = self.negative_sampling(positive_triple, triples_pool)
            sample = (positive_triple, negative_triple)
            if sample not in batch:
                # 这里以dict key的形式存储，当batch比较大时，可以提高not in的计算效率
                batch[sample] = 1

        return batch.keys()

    def negative_sampling(self, positive_triple, triples_pool):
        # 负采样（filter方式，确保负采样得到的三元组不存在于triples_pool中）
        # triples_pool使用dict形式，将(s, r, t)作为key，value任意，可以大幅提高in的计算效率
        negative_triple = positive_triple
        seed = random.random()
        if seed > 0.5:
            # 替换头实体
            while negative_triple in triples_pool:
                negative_triple = (random.randint(0, self.entity_nums - 1), positive_triple[1], positive_triple[2])
        else:
            # 替换尾实体
            while negative_triple in triples_pool:
                negative_triple = (positive_triple[0], positive_triple[1], random.randint(0, self.entity_nums - 1))

        return negative_triple

    def test(self, triples_pool, sample_nums=100):
        test_samples = random.sample(triples_pool, sample_nums)
        mean_rank = hit10 = hit3 = hit1 = i = 0

        for s, r, t in test_samples:
            i += 1
            entity_rank = list()
            for e in range(self.entity_nums):
                entity_rank.append((e, float(self.distance(s, r, e))))
            entity_rank.sort(key=lambda x: x[1])

            rank = 0
            for index in range(len(entity_rank)):
                if entity_rank[index][0] == t:
                    rank = index + 1
                    mean_rank += rank
                    if rank == 1:
                        hit1 += 1
                    if rank <= 3:
                        hit3 += 1
                    if rank <= 10:
                        hit10 += 1
                    break

            # 输出截至当前样本的各项指标
            print("\rsample: {}/{}, rank: {}, mean rank: {}, hit@1: {}, hit@3: {}, hit@10: {}".format(i,
                                                                                                      sample_nums,
                                                                                                      rank,
                                                                                                      round(mean_rank / i, 2),
                                                                                                      round(hit1 / i, 2),
                                                                                                      round(hit3 / i, 2),
                                                                                                      round(hit10 / i, 2)), end='')

        return mean_rank / sample_nums, hit1 / sample_nums, hit3 / sample_nums, hit10 / sample_nums

