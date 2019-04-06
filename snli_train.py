import json
import re
import string
import argparse
import math
import time
import random
import numpy as np
import torch
from torch import optim
from model import DecomposableAttention, SNLIDataset

def train_classifier(model, snl_train_file, snl_test_file, vocab, learning_rate, iterations, oov_embedding, experiment):
    snl_dataset = SNLIDataset.get_features(snl_train_file)
    snl_test = SNLIDataset.get_features(snl_test_file)

    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    total_count = 0
    for epoch in range(iterations):
        cum_loss = 0.0
        start = int(round(time.time() * 1000))
        for batch_idx, features in enumerate(snl_dataset):
            optimizer.zero_grad()
            pair_count = batch_idx + 1
            batch1_start = int(round(time.time() * 1000))
            sent1_feat, sent2_feat, true_label = feat_to_vec(model, features, vocab, oov_embedding, experiment)

            output = model(sent1_feat, sent2_feat)
            loss = model.loss_fun(output.view(1, -1), true_label)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            total_count += 1

            if total_count % ACCURACY_CALC_COUNT == 0:
                accuracy = accuracy_on_dataset(model, snl_test, vocab, oov_embedding, experiment)
                end = int(round(time.time() * 1000))
                took = end - start
                log(epoch, total_count, pair_count, cum_loss, took, accuracy, model)
                start = int(round(time.time() * 1000))
            elif total_count % 100 == 0:
                batch1_end = int(round(time.time() * 1000))
                batch1_took = batch1_end - batch1_start
                log(epoch, total_count, pair_count, cum_loss, batch1_took, 0.0, model)

    accuracy_test = accuracy_on_dataset(model, snl_test, vocab, oov_embedding, experiment)
    print('TEST Accuracy at END:')
    log(1, 1, 1, 1, 1.0, accuracy_test, model)

    accuracy_train = accuracy_on_dataset(model, snl_dataset, vocab, oov_embedding, experiment)
    print('TRAIN Accuracy at END:')
    log(1, 1, 1, 1, 1.0, accuracy_train, model)

def log(epoch, total_count, pair_count, cum_loss, took, accuracy, model):
    if accuracy != 0.0:
        print('%d: %d: loss: %.3f: Test Accuracy: %.5f: epoch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, accuracy, took))
    else:
        print('%d: %d: loss: %.3f: epoch-took: %dmilli' %
              (epoch + 1, total_count, cum_loss / pair_count, took))

def accuracy_on_dataset(model, dataset, vocab, oov_embedding, experiment):
    good = bad = 0.0
    for sentence in dataset:
        sent1_feat, sent2_feat, true_label = feat_to_vec(model, sentence, vocab, oov_embedding, experiment)
        predictions = model.predict(sent1_feat, sent2_feat)

        if predictions == true_label.item():
            good += 1
        else:
            bad += 1

    return good / (good + bad)

def feat_to_vec(model, features, vocab, oov_embedding, experiment):
    sent1_words = np.array(features[0])
    sent2_words = np.array(features[1])
    if experiment in [1, 2]:
        sent1_feat = get_words_indexs_lookup_1(model, sent1_words, vocab)
        sent2_feat = get_words_indexs_lookup_1(model, sent2_words, vocab)
    else:
        sent1_feat = get_words_indexs_lookup_2(model, sent1_words, vocab, oov_embedding)
        sent2_feat = get_words_indexs_lookup_2(model, sent2_words, vocab, oov_embedding)

    true_label = torch.tensor([model.labels[features[2]]])

    if USE_CUDA and torch.cuda.is_available():
        true_label = true_label.cuda()

    return sent1_feat, sent2_feat, true_label

def get_words_indexs_lookup_2(model, words, vocab, oov_embedding):
    ids = list()
    oovs = list()
    oovs_words = None
    for word in words:
        # word = word[0] # Needed for the way dataloader saves the data
        cleaned = clean_word(word)
        if word in vocab:
            ids.append(vocab[word])
        elif cleaned in vocab:
            ids.append(vocab[cleaned])
        else:
            oovs.append(oov_embedding[random.randint(0, 99)])

    lookup = torch.tensor(ids, dtype=torch.long)

    if len(oovs) > 0:
        oovs_words = torch.from_numpy(np.vstack(oovs)).float()

    if USE_CUDA and torch.cuda.is_available():
        lookup = lookup.cuda()
        if len(oovs) > 0:
            oovs_words = oovs_words.cuda()

    sent = model.embed(lookup)
    if oovs_words is not None:
        sent = torch.cat((sent, oovs_words), dim=0)

    if model.project_embedd is not None:
        sent = model.project_embedd(sent)

    return sent

def get_words_indexs_lookup_1(model, words, vocab):
    ids = list()
    for word in words:
        cleaned = clean_word(word)
        if word in vocab:
            ids.append(vocab[word])
        elif cleaned in vocab:
            ids.append(vocab[cleaned])
        else:
            ids.append(vocab['{UNK}'])

    lookup = torch.tensor(ids, dtype=torch.long)
    if USE_CUDA and torch.cuda.is_available():
        lookup = lookup.cuda()

    sent = model.embed(lookup)

    if model.project_embedd is not None:
        sent = model.project_embedd(sent)

    return sent

def read_data(path_to_json):
    with open(path_to_json, 'r') as json_snl_in:
        snl_raw = json_snl_in.readlines()

    snl_json_obj = list()
    for line in snl_raw:
        snl_json_obj.append(json.loads(line))

    return snl_json_obj


def clean_word(word):
    # delete the punctuation in sentence.
    clean = re.sub('[' + string.punctuation + ']', '', word)
    return clean


def run_train(experiment, train_file, test_file, glove_file, vocab_file, learning_rate, iterations, model_out):
    snl_train_file = read_data(train_file)
    snl_test_file = read_data(test_file)

    with open(glove_file, 'rb') as glove_pickle_file:
        glove_embedd = np.load(glove_pickle_file)

    with open(vocab_file, 'r') as vocab_file:
        vocab = json.load(vocab_file)

    oov_embedding = None
    # Adding the UNK vector to the embedding matrix (only for experiment 1-2)
    if experiment  in [1, 2]:
        glove_embedd = np.append(glove_embedd, np.random.uniform(low=EMBED_VALUE_LOW, high=EMBED_VALUE_HIGH, size=(1,EMBED_SIZE)), axis=0)
        vocab['{UNK}'] = len(vocab)
    elif experiment in [3]:
        oov_embedding = np.random.uniform(low=EMBED_VALUE_LOW, high=EMBED_VALUE_HIGH, size=(100, MODEL_SIZE))
    else:
        oov_embedding = np.random.uniform(low=EMBED_VALUE_LOW, high=EMBED_VALUE_HIGH, size=(100, EMBED_SIZE))

    labels = {"contradiction": 0, "entailment": 1, "neutral": 2}

    snli = DecomposableAttention(MODEL_SIZE, MODEL_SIZE, MODEL_SIZE, labels, glove_embedd, experiment)

    if USE_CUDA and torch.cuda.is_available():
        snli.cuda()

    train_classifier(snli, snl_train_file, snl_test_file, vocab, learning_rate, iterations, oov_embedding, experiment)

    if model_out:
        torch.save(snli, model_out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', type=str, help='experiment number (1-7)', required=True)
    parser.add_argument('--trainFile', type=str, help='train file', required=True)
    parser.add_argument('--testFile', type=str, help='test file', required=True)
    parser.add_argument('--modelFile', type=str, help='model output location', required=False)
    parser.add_argument('--gloveFile', type=str, help='preprocessed glove file', required=False)
    parser.add_argument('--vocabFile', type=str, help='preprocessed vocab file', required=True)
    parser.add_argument('--lr', type=str, help='learning rate', required=True)
    parser.add_argument('--iter', type=str, help='num of iterations', required=True)
    parser.add_argument('--cuda', type=str, help='use cuda device', required=False)

    args = parser.parse_args()

    experiment_ = int(args.expr)
    train_file_ = args.trainFile
    test_file_ = args.testFile
    glove_file_ = args.gloveFile
    vocab_file_ = args.vocabFile
    model_out_ = args.modelFile
    learning_rate_ = float(args.lr)
    iterations_ = int(args.iter)

    if experiment_ in [2, 3]:
        EMBED_SIZE = 300
        MODEL_SIZE = 200
    else:
        EMBED_SIZE = 300
        MODEL_SIZE = 300

    # WORD_EMBED_VALUE_LOW = -1/(2 * EMBED_SIZE)
    # WORD_EMBED_VALUE_HIGH = 1/(2 * EMBED_SIZE)
    EMBED_VALUE_HIGH = math.sqrt(6) / math.sqrt(EMBED_SIZE)
    EMBED_VALUE_LOW = -math.sqrt(6) / math.sqrt(EMBED_SIZE)
    ACCURACY_CALC_COUNT = 25000
    USE_CUDA = args.cuda in ['True', 'true', 'yes', 'Yes']

    if USE_CUDA:
        print(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)

    random.seed(1)
    np.random.seed(1)

    run_train(experiment_, train_file_, test_file_, glove_file_, vocab_file_, learning_rate_, iterations_, model_out_)