import json

from markov_chain import MarkovChainBuilder

if __name__ == '__main__':
    with open('datasets/all_names.json', 'rt') as fp:
        names = json.load(fp)
    char_set = {c for n in names for c in n}
    assert '^' not in char_set
    assert '$' not in char_set

    seq_lens = [1, 2, 3, 4, 5]
    for seq_len in seq_lens:
        builder = MarkovChainBuilder('^', '$', seq_len, True)
        builder.process_strings(names)
        markov_chain = builder.compile()

        with open(f'markov_chains/markov_chain_seq_len_{seq_len}.json', 'wt') as fp:
            json.dump(markov_chain.to_json(), fp, separators=(',', ':'))
