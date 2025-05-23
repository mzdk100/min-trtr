from torch import tensor, utils
from torch.nn.utils.rnn import pad_sequence


# 定义数据集类，用于加载源语言和目标语言的句子
class TranslationDataset(utils.data.Dataset):
    @staticmethod
    def get_raw_data(train):
        t = 'train' if train else 'test'
        with open('data/dataset_%s.txt' % t, 'r', encoding='utf-8') as f:
            lines = [i.strip().split('\t') for i in f]
        source_sentences = [i[0] for i in lines]  # 源语言句子列表
        target_sentences = [i[1] for i in lines]  # 目标语言句子列表
        return source_sentences, target_sentences

    def __init__(self, source_sentences, target_sentences, src_vocab, tgt_vocab):
        """source_sentences 和 target_sentences 都需进行分词"""

        self.source_sentences, self.target_sentences = source_sentences, target_sentences
        self.src_vocab = src_vocab  # 源语言词汇表
        self.tgt_vocab = tgt_vocab  # 目标语言词汇表

    def __len__(self):
        return len(self.source_sentences)  # 返回数据集中句子的数量

    def __getitem__(self, idx):
        # 将源语言和目标语言的句子转换为词汇表中的索引
        src = [self.src_vocab[word] for word in self.source_sentences[idx]]
        tgt = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word] for word in self.target_sentences[idx]] + [self.tgt_vocab['<eos>']]
        return tensor(src), tensor(tgt)  # 返回源句子和目标句子的索引张量

# 定义collate_fn函数，用于在批处理中对序列进行填充
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)  # 将批次中的源和目标句子分开
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)  # 对源句子进行填充
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)  # 对目标句子进行填充
    return src_batch, tgt_batch  # 返回填充后的源和目标句子张量
