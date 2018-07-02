import torch
import torch.nn.functional as functional
from torch import autograd, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.data import data


class LSTMFeatureExtractor(nn.Module):
    def __init__(self,
                 word_vocab, postag_vocab, position1_vocab, position2_vocab,
                 num_layers,
                 hidden_size,
                 dropout):
        super(LSTMFeatureExtractor, self).__init__()
        self.num_layers = num_layers

        self.hidden_size = hidden_size // 2
        self.n_cells = self.num_layers * 2

        self.word_emb = word_vocab.init_embed_layer()
        self.postag_emb = postag_vocab.init_embed_layer()
        self.position1_emb = position1_vocab.init_embed_layer()
        self.position2_emb = position2_vocab.init_embed_layer()

        self.input_size = word_vocab.emb_size + postag_vocab.emb_size + position1_vocab.emb_size + position2_vocab.emb_size

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=num_layers, dropout=dropout, bidirectional=True)

        self.attn = DotAttentionLayer(hidden_size)

    def forward(self, x2, x1):
        tokens, postag, positions1, positions2, e1_token, e2_token = x2
        _, _, _, _, _, _, lengths, sort_idx = x1

        lengths_list = lengths.tolist()
        batch_size = tokens.size(0)

        tokens = self.word_emb(tokens)  # (bz, seq, emb)
        postag = self.postag_emb(postag)
        positions1 = self.position1_emb(positions1)
        positions2 = self.position2_emb(positions2)

        embeds = torch.cat((tokens, postag, positions1, positions2), 2)  # (bz, seq, ?)

        packed = pack_padded_sequence(embeds, lengths_list, batch_first=True)
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = embeds.new(*state_shape)
        output, (ht, ct) = self.rnn(packed, (h0, c0))

        unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
        return self.attn((unpacked_output, lengths))


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if torch.cuda.is_available():
            idxes = idxes.cuda(data.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output


class MLP(nn.Module):
    def __init__(self, context_feature_size, relation_vocab, entity_type_vocab, entity_vocab,  tok_num_betw_vocab,
                                         et_num_vocab):
        super(MLP, self).__init__()

        self.entity_type_emb = entity_type_vocab.init_embed_layer()

        self.entity_emb = entity_vocab.init_embed_layer()

        self.dot_att = DotAttentionLayer(entity_vocab.emb_size)

        self.tok_num_betw_emb = tok_num_betw_vocab.init_embed_layer()

        self.et_num_emb = et_num_vocab.init_embed_layer()

        self.input_size = context_feature_size + 2 * entity_type_vocab.emb_size + 2 * entity_vocab.emb_size + \
                          tok_num_betw_vocab.emb_size + et_num_vocab.emb_size

        self.linear = nn.Linear(self.input_size, relation_vocab.vocab_size, bias=False)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, hidden_features, x2, x1):
        _, _, _, _, e1_token, e2_token = x2
        e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, lengths, _ = x1

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        output = self.linear(x)

        return output

    def loss(self, by, y_pred):

        return self.criterion(y_pred, by)


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 word_vocab, postag_vocab, position1_vocab, position2_vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.word_emb = word_vocab.init_embed_layer()
        self.postag_emb = postag_vocab.init_embed_layer()
        self.position1_emb = position1_vocab.init_embed_layer()
        self.position2_emb = position2_vocab.init_embed_layer()

        self.input_size = word_vocab.emb_size+postag_vocab.emb_size+position1_vocab.emb_size+position2_vocab.emb_size

        self.hidden_size = hidden_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, self.input_size)) for K in kernel_sizes])



        # at least 1 hidden layer so that the output size is hidden_size
        assert num_layers > 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                                      nn.Linear(len(kernel_sizes) * kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))


            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, x2, x1):
        tokens, postag, positions1, positions2, e1_token, e2_token = x2

        tokens = self.word_emb(tokens)  # (bz, seq, emb)
        postag = self.postag_emb(postag)
        positions1 = self.position1_emb(positions1)
        positions2 = self.position2_emb(positions2)

        embeds = torch.cat((tokens, postag, positions1, positions2), 2) # (bz, seq, ?)

        # conv
        embeds = embeds.unsqueeze(1)  # batch_size, 1, seq_len, emb_size

        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)