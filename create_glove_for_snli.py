import argparse
import json
import numpy as np

from snli_train import read_data, clean_word


def get_all_words(snl_file):
    all_words = list()
    for sent in snl_file:
        sent_words = list()
        sent_words.extend(sent['sentence1'].split())
        sent_words.extend(sent['sentence2'].split())
        all_words.extend(word for word in sent_words)
        all_words.extend([clean_word(word) for word in sent_words])

    print('Added a total of ' + str(len(list(all_words))) + ' words to all_words')
    return all_words

def load_glove_for_vocab(snl_vocabulary, glove_filename):
    embedd = list()
    final_vocab = dict()
    with open(glove_filename, 'r') as glove:
        index = 0
        for line in glove.readlines():
            row = line.strip().split(' ')
            word = row[0].strip()
            if word in final_vocab:
                continue
            if word in snl_vocabulary:
                embedd.append(np.array(row[1:], dtype=float))
                final_vocab[word] = index
                index += 1
    print('Done exporting embeddings, Total of-' + str(len(embedd)) + ' embeddings exported')
    return final_vocab, np.array(embedd)


def create_glove_subset(snl_file_list, output_embedd, output_vocab):
    print('get all snli words....')
    all_words = list()
    for snl_data_file in snl_file_list:
        all_words.extend(get_all_words(snl_data_file))

    all_words = set(all_words)
    print('Start generating embedding and vocab...')
    final_vocab, embedding_vocab_matrix = load_glove_for_vocab(all_words, glove_file)

    print('In total vocab size is: ' + str(len(final_vocab)))

    print('Saving embedding numpy matrix...')
    with open(output_embedd, 'wb') as f_embedd:
        np.save(f_embedd, embedding_vocab_matrix)

    print('Saving json vocab matrix...')
    with open(output_vocab, 'w') as f_vocab:
        json.dump(final_vocab, f_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainFile', type=str, help='train file', required=True)
    parser.add_argument('--testFile', type=str, help='test file', required=True)
    parser.add_argument('--devFile', type=str, help='dev file', required=True)
    parser.add_argument('--gloveFile', type=str, help='Glove downloaded file', required=True)
    parser.add_argument('--outputEmbed', type=str, help='location to save embed output', required=True)
    parser.add_argument('--outputVocab', type=str, help='location to save vocab', required=True)

    args = parser.parse_args()

    snl_train = read_data(args.trainFile)
    snl_dev = read_data(args.devFile)
    snl_test = read_data(args.testFile)

    snl_file_list = [snl_train, snl_dev, snl_test]

    glove_file = args.gloveFile
    output_embed = args.outputEmbed
    output_vocab = args.outputVocab
    create_glove_subset(snl_file_list, output_embed, output_vocab)
    print("Process Done!!!")
