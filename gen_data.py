from glue_utils import InputExample
import sys
import torch

# convert .pth file to .txt file (use for generating adversarial examples in text format)
def create_file(mode):
    examples = torch.load(f'{sys.argv[1]}_adv/{mode}-examples.pth')
    with open(f'{sys.argv[1]}_adv/{mode}.txt', 'w') as f:
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
    for mode in ['train', 'dev', 'test']:
        create_file(mode)
