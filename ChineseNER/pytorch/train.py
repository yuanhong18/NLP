# coding=utf-8
import pickle, argparse
import numpy as np
import pandas as pd
from Batch import BatchGenerator
#### read data
with open('../data/Bosondata.pkl', 'rb') as inp:
	word2id = pickle.load(inp)
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	id2tag = pickle.load(inp)
	x_train = pickle.load(inp)
	y_train = pickle.load(inp)
	x_test = pickle.load(inp)
	y_test = pickle.load(inp)
	x_valid = pickle.load(inp)
	y_valid = pickle.load(inp)
print("train len:",len(x_train))
print("test len:",len(x_test))
print("valid len", len(x_valid))

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs 
from BiLSTM_CRF import BiLSTM_CRF
from resultCal import calculate

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 100

tag2id[START_TAG]=len(tag2id)
tag2id[STOP_TAG]=len(tag2id)

def main(args):
    if args.train:
        # trian
        # data generator
        batch_size = 32
        batch_nums = (int(len(x_train) / batch_size), int(len(x_test) / batch_size))
        data_train = BatchGenerator(x_train, y_train, shuffle=True)
        data_test = BatchGenerator(x_test, y_test, shuffle=False)
        # model
        model = BiLSTM_CRF(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM, batch_size)
        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
        for epoch in range(EPOCHS):
            for batch in range(batch_nums[0]):
                batch += 1
                sentence, tags = data_train.next_batch(batch_size)
                model.zero_grad()

                sentence = torch.tensor(sentence, dtype=torch.long)
                tags = torch.tensor(tags, dtype=torch.long)

                loss = model.neg_log_likelihood(sentence, tags)
                loss.mean().backward()

                optimizer.step()
                if batch % 100 == 0:
                    score, predict = model(sentence)
                    print("epoch: {}, batch {} / bath_nums {}, loss: {}".format(epoch, batch, batch_nums[0], loss.mean()))

                    res = eval(sentence, predict, tags)
                    print("accuracy: {} recall: {} f1: {}".format(res[0], res[1], res[2]))
            for batch in range(batch_nums[1]):
                sentence, tags = data_test.next_batch(batch_size)
                sentence = torch.tensor(sentence, dtype=torch.long)
                score, predict = model(sentence)
                res = eval(sentence, predict, tags)
                if batch % 100 == 0:
                    print("accuracy: {} recall: {} f1: {}".format(res[0], res[1], res[2]))

            path_name = "./model/model" + str(epoch) + ".pkl"
            print(path_name)
            torch.save(model.state_dict(), path_name)
            print("model has been saved")
    elif args.test:
        sents = input("input:")
        max_len = 60

        def X_padding(words):
            ids = list(word2id[words])
            if len(ids) >= max_len:
                return ids[:max_len]
            ids.extend([0] * (max_len - len(ids)))
            return ids

        sentence = [[i for i in sents]]
        df_data = pd.DataFrame({'words': sentence}, index=list(range(len(sentence))))

        df_data['x'] = df_data['words'].apply(X_padding)
        x = np.asarray(list(df_data['x'].values))
        x = torch.tensor(x, dtype=torch.long)
        model = BiLSTM_CRF(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM, 1)
        model.load_state_dict(torch.load("model/model99.pkl", map_location="cpu"))
        score, predict = model(x)
        predict = predict.permute(1, 0)

        print(id2tag)
        for i in range(len(predict)):
            print([sentence[i][j] + "/" + id2tag[predict[i][j].item()] for j in range(len(sentence[i]))])
    else:
        print("please choose right mode!")

# eval fuction
def eval(sentence, predict, tags):
    entityres, entityall = [], []
    entityres = calculate(sentence, predict.permute(1, 0), id2word, id2tag, entityres)
    entityall = calculate(sentence, tags, id2word, id2tag, entityall)

    intersection = []
    acc, recall, f1 = [], [], []
    for i in range(len(entityres)):
        intersection.append([j for j in entityres[i] if j in entityall[i]])
    for i in range(len(intersection)):
        if len(entityres[i]) == 0:
            acc_ = 1.
        else:
            acc_ = float(len(intersection[i]) / len(entityres[i]))
        if len(entityall[i]) == 0:
            recall_ = 1.
        else:
            recall_ = float(len(intersection[i]) / len(entityall[i]))
        if acc_ + recall_ == 0.:
            f1_ = 0.
        else:
            f1_ = (2 * acc_ * recall_) / (acc_ + recall_)
        acc.append(acc_)
        recall.append(recall_)
        f1.append(f1_)
    return np.mean(acc), np.mean(recall), np.mean(f1)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    group = arg.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    args = arg.parse_args()
    main(args)
