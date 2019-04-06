from torch import nn
import torch
import torch.nn.functional as F

import numpy as np
from torch.utils.data import Dataset

class SNLIDataset(Dataset):
    def __init__(self, snl_file):
        self.snl_data = self.get_features(snl_file)

    def __len__(self):
        return len(self.snl_data)

    def __getitem__(self, idx):
        return self.snl_data[idx]

    @staticmethod
    def get_features(snl_jsons_list):
        """
        Return list of tuples (Sentence1[300d vector for each sentence word],
            Sentence2[300d vector for each sentence word], Label)
        """
        features = list()
        for snl_json_obj in snl_jsons_list:
            sentence1_words = snl_json_obj['sentence1'].split()
            sentence2_words = snl_json_obj['sentence2'].split()
            gold_label = snl_json_obj['gold_label']
            if gold_label != '-':  # don't considerate '-' labels
                features.append((sentence1_words, sentence2_words, gold_label))

        return features


class DecomposableAttention(nn.Module):

    def __init__(self, f_in_dim, f_hid_dim, f_out_dim, labels, embeddings, experiment, embedd_dim = 300):
        super(DecomposableAttention, self).__init__()

        self.project_embedd = None

        self.F = DecomposableAttention.get_sequential(f_in_dim, f_hid_dim, f_out_dim, experiment)
        self.G = DecomposableAttention.get_sequential(2 * f_in_dim, f_hid_dim, f_out_dim, experiment)
        self.H = DecomposableAttention.get_sequential(2 * f_in_dim, f_hid_dim, f_out_dim, experiment)

        self.last_layer = nn.Linear(f_out_dim, len(labels))

        if experiment in [2, 3]:
            # PCA Project of glove from 300d to 200d
            self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.embed.weight = nn.Parameter(torch.from_numpy(self.get_projected_matrix(embeddings, f_in_dim)))
            self.embed.float()
        else:
            # Glove embedding
            self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))
            self.embed.float()

            if experiment in [1, 4]:
                # Linear Project 300d to 200d
                self.project_embedd = nn.Linear(embedd_dim, f_in_dim)
                self.embed.weight.requires_grad = False

            self.labels = labels

            self.loss_fun = nn.CrossEntropyLoss()

    @staticmethod
    def get_projected_matrix(np_embedd, new_size):
        x_prop = (np_embedd - np.mean(np_embedd)) / np.std(np_embedd)
        sigma = np.dot(np.transpose(np_embedd), np_embedd) / np_embedd.shape[0] * np_embedd.shape[1]
        u, s, vh = np.linalg.svd(sigma, full_matrices=True)
        reduce = u[:, :new_size]
        np_embedd_reduce = np.dot(x_prop, reduce)
        return np_embedd_reduce

    @staticmethod
    def get_sequential(ind, hidd, outd, experiment):
        if experiment == 6 :
            return nn.Sequential(nn.Dropout(0.2), nn.Linear(ind, hidd), nn.ReLU(), nn.Linear(hidd, outd), nn.ReLU())
        else:
            return nn.Sequential(nn.Linear(ind, hidd), nn.ReLU(), nn.Linear(hidd, outd), nn.ReLU())

    def forward(self, sent1_feat, sent2_feat):
        # Attentd
        attend_out1 = self.F(sent1_feat)
        attend_out2 = self.F(sent2_feat)

        eij1 = torch.mm(attend_out1, attend_out2.t())
        eij2 = eij1.t()
        eij1_soft = F.softmax(eij1, dim=1)
        eij2_soft = F.softmax(eij2, dim=1)

        alpha = torch.mm(eij2_soft, sent1_feat)
        beta = torch.mm(eij1_soft, sent2_feat)

        # compare
        compare_i = torch.cat((sent1_feat, beta), dim=1)
        compare_j = torch.cat((sent2_feat, alpha), dim=1)
        v1_i = self.G(compare_i)
        v2_j = self.G(compare_j)

        # Aggregate (3.3)
        v1_sum = torch.sum(v1_i, dim=0)
        v2_sum = torch.sum(v2_j, dim=0)

        output_tolast = self.H(torch.cat((v1_sum, v2_sum))).view(1, -1)

        output = self.last_layer(output_tolast)

        return output