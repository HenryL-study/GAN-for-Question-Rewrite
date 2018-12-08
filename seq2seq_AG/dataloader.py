import numpy as np


class Data_loader():
    def __init__(self, batch_size, ques_len, ans_len):
        self.batch_size = batch_size
        self.token_stream = []
        self.num_stream = []
        self.MAX_Q_LEN = ques_len
        self.MAX_ANS_LEN = ans_len

    def create_batches(self, ques_file, ques_len_file, ans_file, ans_len_file):
        self.token_stream = []
        self.num_stream = []
        self.ans_token_stream = []
        self.ans_num_stream = []
        
        with open(ques_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.MAX_Q_LEN:
                    self.token_stream.append(parse_line)
        with open(ques_len_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                self.num_stream = [int(x) for x in line]
        with open(ans_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.MAX_ANS_LEN:
                    self.ans_token_stream.append(parse_line)
        with open(ans_len_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                self.ans_num_stream = [int(x) for x in line]

        self.num_batch = int(7999 / self.batch_size)
        self.num_test_batch = int(self.num_batch * 0.2)
        self.total_batch = self.num_batch
        self.num_batch -= self.num_test_batch

        self.token_stream = self.token_stream[:self.total_batch * self.batch_size]
        self.num_stream = self.num_stream[:self.total_batch * self.batch_size]
        self.ans_token_stream = self.ans_token_stream[:self.total_batch * self.batch_size]
        self.ans_num_stream = self.ans_num_stream[:self.total_batch * self.batch_size]

        self.sequence_batch = np.split(np.array(self.token_stream), self.total_batch, 0)
        self.sequence_len_batch = np.split(np.array(self.num_stream), self.total_batch)
        self.ans_batch = np.split(np.array(self.ans_token_stream), self.total_batch, 0)
        self.ans_len_batch = np.split(np.array(self.ans_num_stream), self.total_batch)

        print("question batch: ", len(self.sequence_batch), self.sequence_batch[0].shape)
        print("ans batch: ", len(self.ans_batch), self.ans_batch[0].shape)
        self.pointer = 0
        self.test_pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        seq_len = self.sequence_len_batch[self.pointer]
        ans = self.ans_batch[self.pointer]
        ans_len = self.ans_len_batch[self.pointer]

        self.pointer = (self.pointer + 1) % self.num_batch
        return ret, seq_len, ans, ans_len
    
    def next_test_batch(self):
        ret = self.sequence_batch[self.test_pointer + self.num_batch]
        seq_len = self.sequence_len_batch[self.test_pointer + self.num_batch]
        ans = self.ans_batch[self.test_pointer + self.num_batch]
        ans_len = self.ans_len_batch[self.test_pointer + self.num_batch]

        self.test_pointer = (self.test_pointer + 1) % self.num_test_batch
        return ret, seq_len, ans, ans_len

    def reset_pointer(self):
        self.pointer = 0
        self.test_pointer = 0
