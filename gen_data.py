from glue_utils import InputExample
import sys
import torch

def create_file(mode):
    examples = torch.load(f'adv-laptop/{mode}-examples.pth')
    with open(f'adv-laptop/adv-{mode}.txt', 'w') as f:
        for example in examples:
            words = example.text_a.split(' ')
            line = []
            labels = example.label
            for word, label in zip(words, labels):
                term = label
                if label != 'O':
                    term = 'T' + label[-4:]
                line.append(f'{word}={term}')
            line = example.text_a + '####' + ' '.join(line) + '\n'
            f.write(line)

if __name__ == "__main__":
    create_file('test' if len(sys.argv) == 1 else sys.argv[1])
