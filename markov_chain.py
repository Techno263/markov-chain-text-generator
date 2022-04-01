import numpy as np
import random


MAX_LENGTH = 30


def get_char_sequences(seq_len, characters):
    if seq_len < 1:
        raise Exception('seq_len must be greater than 0')
    if seq_len == 1:
        return characters
    output = get_char_sequences(seq_len - 1, characters)
    output = output + [
        s + c
        for s in output
        for c in characters
    ]
    return output


class MarkovChainState:

    def __init__(self, state, transitions, probs):
        self.state = state
        self.transitions = transitions
        self.probs = probs

    def get_next(self):
        return np.random.choice(self.transitions, p=self.probs)

    def to_json(self):
        return {
            'state': self.state,
            'transitions': self.transitions.tolist(),
            'probs': self.probs.tolist()
        }

    @staticmethod
    def from_json(obj):
        return MarkovChainState(
            obj['state'],
            np.array(obj['transitions']),
            np.array(obj['probs'])
        )


class MarkovChain:

    def __init__(self, states, start, end, seq_len):
        self.states = states
        self.start = start
        self.end = end
        self.seq_len = seq_len

    def _generate_string(self):
        output = []
        current_state = self.states[self.start]
        for i in range(MAX_LENGTH):
            transition = current_state.get_next()
            if transition == self.end:
                break
            output.append(transition)
            state = ''.join(output[-self.seq_len:])
            current_state = self.states[state]
        return ''.join(output)

    def generate_string(self, min_len):
        output = self._generate_string()
        while len(output) < min_len:
            output = self._generate_string()
        return output

    def to_json(self):
        return {
            'seq_len': self.seq_len,
            'start': self.start,
            'end': self.end,
            'states': [s.to_json() for s in self.states.values()]
        }

    @staticmethod
    def from_json(obj):
        return MarkovChain(
            {s['state']: MarkovChainState.from_json(s) for s in obj['states']},
            obj['start'],
            obj['end'],
            obj['seq_len']
        )


class MarkovChainBuilderState:

    def __init__(self, state):
        self.state = state
        self.transitions = {}

    def increment_transision(self, character):
        if character not in self.transitions:
            self.transitions[character] = 1
        else:
            self.transitions[character] += 1

    def compile(self):
        transitions, counts = map(np.array, zip(*self.transitions.items()))
        count_sum = np.sum(counts)
        if count_sum == 0:
            probs = np.full(counts.shape, 1 / counts.size)
        else:
            probs = counts / np.sum(counts)
        return MarkovChainState(self.state, transitions, probs)

    def to_json(self):
        return {
            'state': self.state,
            'transitions': self.transitions
        }

    @staticmethod
    def from_json(obj):
        mcbs = MarkovChainBuilderState(obj['state'])
        mcbs.transitions = obj['transitions']
        return mcbs


class MarkovChainBuilder:

    def __init__(self, start, end, seq_len, nested_seq=False):
        assert len(start) == 1
        assert len(end) == 1
        assert start != end
        assert seq_len > 0

        self.start = start
        self.end = end
        self.seq_len = seq_len
        self.nested_seq = nested_seq
        self.states = {}

    def get_state(self, state):
        if state in self.states:
            return self.states[state]
        s = MarkovChainBuilderState(state)
        self.states[state] = s
        return s

    def process_string(self, string):
        assert self.start not in string
        assert self.end not in string

        self.get_state(self.start).increment_transision(string[0])
        state = string[0]
        for i in range(1, min(self.seq_len, len(string))):
            t = string[i]
            self.get_state(state).increment_transision(t)
            if self.nested_seq:
                for j in range(i - 1, 0, -1):
                    self.get_state(state[-j:]).increment_transision(t)
            state = (state + t)[-self.seq_len:]
        for t in string[self.seq_len:]:
            self.get_state(state).increment_transision(t)
            if self.nested_seq:
                for j in range(self.seq_len - 1, 0, -1):
                    self.get_state(state[-j:]).increment_transision(t)
            state = (state + t)[-self.seq_len:]
        self.get_state(state).increment_transision(self.end)
        if self.nested_seq:
            for i in range(self.seq_len - 1, 0, -1):
                self.get_state(state[-i:]).increment_transision(self.end)

    def process_strings(self, strings):
        for string in strings:
            self.process_string(string)

    def compile(self):
        states = {c: s.compile() for c, s in self.states.items()}
        return MarkovChain(states, self.start, self.end, self.seq_len)

    def to_json(self):
        return {
            'start': self.start,
            'end': self.end,
            'seq_len': self.seq_len,
            'nested_seq': self.nested_seq,
            'states': [s.to_json() for s in self.states.values()]
        }

    @staticmethod
    def from_json(obj):
        mcb = MarkovChainBuilder(obj['start'], obj['end'], obj['seq_len'], obj['nested_seq'])
        mcb.states = {
            s['state']: MarkovChainBuilderState.from_json(s)
            for s in obj['states']
        }
        return mcb
