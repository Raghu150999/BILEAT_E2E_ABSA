import torch
import torch.nn as nn
from seq_utils import *


class CRF(nn.Module):
    # borrow the code from 
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
    def __init__(self, num_tags, constraints=None, include_start_end_transitions=None):
        """

        :param num_tags:
        :param constraints:
        :param include_start_end_transitions:
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.include_start_end_transitions = include_start_end_transitions
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        constraint_mask = torch.Tensor(self.num_tags+2, self.num_tags+2).fill_(1.)
        if include_start_end_transitions:
            self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        # register the constraint_mask
        self.constraint_mask = nn.Parameter(constraint_mask, requires_grad=False)
        self.reset_parameters()

    def forward(self, inputs, tags, mask=None):
        """

        :param inputs: (bsz, seq_len, num_tags), logits calculated from a linear layer
        :param tags: (bsz, seq_len)
        :param mask: (bsz, seq_len), mask for the padding token
        :return:
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def reset_parameters(self):
        """
        initialize the parameters in CRF
        :return:
        """
        nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            nn.init.normal_(self.start_transitions)
            nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits, mask):
        """

        :param logits: emission score calculated by a linear layer, shape: (batch_size, seq_len, num_tags)
        :param mask:
        :return:
        """
        bsz, seq_len, num_tags = logits.size()
        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for t in range(1, seq_len):
            # iteration starts from 1
            emit_scores = logits[t].view(bsz, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(bsz, num_tags, 1)

            # calculate the likelihood
            inner = broadcast_alpha + emit_scores + transition_scores

            # mask the padded token when met the padded token, retain the previous alpha
            alpha = (logsumexp(inner, 1) * mask[t].view(bsz, 1) + alpha * (1 - mask[t]).view(bsz, 1))
        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(self, logits, tags, mask):
        """
        calculate the likelihood for the input tag sequence
        :param logits:
        :param tags: shape: (bsz, seq_len)
        :param mask: shape: (bsz, seq_len)
        :return:
        """
        bsz, seq_len, _ = logits.size()

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for t in range(seq_len-1):
            current_tag, next_tag = tags[t], tags[t+1]
            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[t].gather(1, current_tag.view(bsz, 1)).squeeze(1)

            score = score + transition_score * mask[t+1] + emit_score * mask[t]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, bsz)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def viterbi_tags(self, logits, mask):
        """

        :param logits: (bsz, seq_len, num_tags), emission scores
        :param mask:
        :return:
        """
        _, max_seq_len, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = (
                self.transitions * self.constraint_mask[:num_tags, :num_tags] +
                -10000.0 * (1 - self.constraint_mask[:num_tags, :num_tags])
        )

        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                    self.start_transitions.detach() * self.constraint_mask[start_tag, :num_tags].data +
                    -10000.0 * (1 - self.constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                    self.end_transitions.detach() * self.constraint_mask[:num_tags, end_tag].data +
                    -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = (-10000.0 *
                                                 (1 - self.constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_len + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            # perform viterbi decoding sample by sample
            seq_len = torch.sum(prediction_mask)
            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(seq_len + 1), :num_tags] = prediction[:seq_len]
            # And at the last timestep we must have the END_TAG
            tag_sequence[seq_len + 1, end_tag] = 0.
            viterbi_path = viterbi_decode(tag_sequence[:(seq_len + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append(viterbi_path)
        return best_paths