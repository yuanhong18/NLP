import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
def argmax(vec):
    # return the argmax as a python int
    item, idx = torch.max(vec, 1)
    return item, idx


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, batch_size):
    max_score = argmax(vec)[0]
    max_score_broadcast = max_score.view(batch_size, -1).expand(batch_size, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(2, self.batch_size, self.hidden_dim // 2),
                torch.zeros(2, self.batch_size, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.batch_size, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(self.batch_size, -1).expand(self.batch_size, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).expand(self.batch_size, self.tagset_size)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var, self.batch_size).view(self.batch_size, 1))
            forward_var = torch.cat(alphas_t, 1).view(self.batch_size, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(1, -1).expand(self.batch_size, self.tagset_size)
        alpha = log_sum_exp(terminal_var, self.batch_size)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).permute(1,0,2)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(self.batch_size)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).expand(self.batch_size, 1), tags], dim=1)
        for i, feat in enumerate(feats):
            index_x = tags[:, i+1].long()
            index_y = tags[:, i].long()
            trans_x = torch.index_select(self.transitions, 0, index_x)
            trans_y = torch.gather(trans_x, dim=1, index=index_y.view(self.batch_size, -1)).view(self.batch_size)
            emit = torch.gather(feat, dim=1, index=index_x.view(self.batch_size, -1)).view(self.batch_size)
            score = score + trans_y + emit
        index_x = torch.tensor(self.tag_to_ix[STOP_TAG]).long().expand(self.batch_size)
        index_y = tags[:, -1].long()
        trans_x = torch.index_select(self.transitions, 0, index_x)
        trans_y = torch.gather(trans_x, dim=1, index=index_y.view(self.batch_size, -1)).view(self.batch_size)
        score = score + trans_y
        return score

    def _viterbi_decode(self, feats, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((batch_size, self.tagset_size), -10000.)
        init_vvars[:, self.tag_to_ix[START_TAG]] = 0.

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag].view(1, -1).expand(batch_size, self.tagset_size)
                best_tag = argmax(next_tag_var)
                bptrs_t.append(best_tag[1].view(batch_size, 1))
                viterbivars_t.append(best_tag[0].view(batch_size, 1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = torch.cat(viterbivars_t, 1) + feat
            backpointers.append(torch.cat(bptrs_t, 1))

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(1, -1).expand(batch_size, self.tagset_size)
        best_tag = argmax(terminal_var)

        # Follow the back pointers to decode the best path.
        best_path = [best_tag[1]]
        best_tag_id = best_tag[1].view(batch_size, -1)
        for bptrs_t in reversed(backpointers):
            best_tag_id = torch.gather(bptrs_t, dim=1, index=best_tag_id).view(batch_size, -1)
            best_path.append(best_tag_id.view(batch_size))
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert  torch.eq(start.view(batch_size), torch.tensor(self.tag_to_ix[START_TAG]).expand(batch_size)).sum() == batch_size  # Sanity check
        best_path.reverse()
        return best_tag[0], torch.stack(best_path, dim=0)


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
