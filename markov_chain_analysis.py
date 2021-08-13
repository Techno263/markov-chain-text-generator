import json

from markov_chain import MarkovChain

if __name__ == '__main__':
    markov_chains = [
        'markov_chains/markov_chain_seq_len_1.json',
        'markov_chains/markov_chain_seq_len_2.json',
        'markov_chains/markov_chain_seq_len_3.json',
        'markov_chains/markov_chain_seq_len_4.json',
        'markov_chains/markov_chain_seq_len_5.json'
    ]
    with open('datasets/all_names.json', 'rt') as fp:
        names = json.load(fp)
    name_set = set(names)
    gen_count = 10000
    for mc in markov_chains:
        with open(mc, 'rt') as fp:
            markov_chain = MarkovChain.from_json(json.load(fp))
        gen_names = [markov_chain.generate_string(2) for _ in range(gen_count)]
        intersection = name_set.intersection(set(gen_names))
        print()
        print(mc)
        print(len(intersection) / gen_count)
        print(sorted(gen_names[:100]))
    print()
