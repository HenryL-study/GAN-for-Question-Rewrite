import numpy as np


class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []
        self.num_stream = []
        self.MAX_SEQ_LEN = 28 #change every time

    def create_batches(self, data_file, data_len_file):
        self.token_stream = []
        self.num_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.MAX_SEQ_LEN:
                    self.token_stream.append(parse_line)
        with open(data_len_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                self.num_stream = [int(x) for x in line]

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.num_stream = self.num_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.sequence_len_batch = np.split(np.array(self.num_stream), self.num_batch)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        seq_len = self.sequence_len_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret, seq_len

    def reset_pointer(self):
        self.pointer = 0

#TODO change to given actual length
class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.MAX_SEQ_LEN = 28 #change every time

    def load_train_data(self, positive_file, positive_len_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        positive_examples_len = []
        negative_examples_len = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(data_len_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                positive_examples_len = [int(x) for x in line]        
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.MAX_SEQ_LEN:
                    negative_examples.append(parse_line)
                    negative_examples_len.append(self.MAX_SEQ_LEN)
                else:
                    print("Problem aroused! need change.")
        self.sentences = np.array(positive_examples + negative_examples)
        self.sentences_len = np.concatenate((np.array(positive_examples_len), np.array(negative_examples_len)))

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.sentences_len = self.sentences_len[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_len = self.sentences_len[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.sequences_len_batches = np.split(self.sentences_len, self.num_batch)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer], self.sequences_len_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

