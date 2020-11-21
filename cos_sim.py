import os
import numpy as np

COS_SIM_PATH = "./cache/cos_sim_counter_fitting.npy"

class CosSim:
    def __init__(self, embedding_path='counter-fitted-vectors.txt'):
        if not os.path.exists("./cache"):
            os.mkdir("./cache")
        if not os.path.exists(COS_SIM_PATH):
            embeddings = []
            with open(embedding_path, 'r') as ifile:
                for line in ifile:
                    embedding = [float(num) for num in line.strip().split()[1:]]
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.asarray(embeddings / norm, "float32")
            np.save((COS_SIM_PATH), embeddings)
        self.embedding = np.load(COS_SIM_PATH)
        self.cache = {}
        self.vocab_size = len(self.embedding)
    
    def compute_sim(self, widx1, widx2):
        # if self.cache.get((widx1, widx2)) == None:
        #     self.cache[(widx1, widx2)] = np.dot(self.embedding[widx1], self.embedding[widx2].T)
        # return self.cache[(widx1, widx2)]
        return np.dot(self.embedding[widx1], self.embedding[widx2].T)
    
    def find_sim_words(self, widx, k=10):
        sims = []
        for i in range(self.vocab_size):
            sims.append(self.compute_sim(widx, i))
        return np.array(sims).argsort()[::-1][:k]
        
if __name__ == "__main__":
    csim = CosSim()
    a = 2447
    b = 27066
    print(csim.compute_sim(a, b))
    # print(csim.find_sim_words(3))
    # Build vocab for counter fitting embedding
    idx2word = {}
    word2idx = {}
    with open('./counter-fitted-vectors.txt', 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1
    print(idx2word[a], idx2word[b])