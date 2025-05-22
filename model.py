from torch import nn, zeros, arange, exp, log, tensor, sin, cos, sqrt, matmul, triu, ones, gt, randn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = zeros(max_len, d_model)
        position = arange(0, max_len).float().unsqueeze(1)
        div_term = exp(arange(0, d_model, 2).float() * (-log(tensor(10000.0)) / d_model))
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        assert d_k == d_v
        self.d_k_sqrt = sqrt(tensor(d_k))
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        scores = matmul(Q, K.transpose(-1, -2)) / self.d_k_sqrt  # [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(scores)  # [batch_size, n_heads, len_q, len_k]，对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = matmul(attn, V)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.norm(output + residual)


# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_feed_forward):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_feed_forward, bias=False),
            nn.ReLU(),
            nn.Linear(d_feed_forward, d_model, bias=False)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return self.norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_feed_forward, n_heads, d_k, d_v):
        super(EncoderLayer, self).__init__()
        assert d_k == d_v
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_feed_forward)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        return self.pos_ffn(enc_outputs)  # [batch_size, src_len, d_model]


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, d_feed_forward, n_layers, n_heads, d_k, d_v, dropout, max_len):
        super(TransformerEncoder, self).__init__()
        assert d_k == d_v
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_feed_forward, n_heads, d_k, d_v) for _ in range(n_layers)])

    def forward(self, src, attn_mask):
        """
        :param src: [batch_size, src_len]
        :param attn_mask: [batch_size, src_len, src_len]
        """
        out = self.src_emb(src)  # [batch_size, src_len, d_model]
        out = self.pos_emb(out.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出out作为当前block的输入
            # out: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
            out = layer(out, attn_mask)  # 传入的out其实是input，传入mask矩阵是因为你要做self attention
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_feed_forward, n_heads, d_k, d_v):
        super(DecoderLayer, self).__init__()
        assert d_k == d_v
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_feed_forward)

    def forward(self, tgt, memory, dec_pad_mask, enc_pad_mask):
        """
        :param tgt: [batch_size, tgt_len, d_model]
        :param memory: [batch_size, src_len, d_model]
        :param dec_pad_mask: [batch_size, tgt_len, tgt_len]
        :param enc_pad_mask: [batch_size, tgt_len, src_len]
        """
        out = self.dec_self_attn(tgt, tgt, tgt, dec_pad_mask)  # [batch_size, tgt_len, d_model]，这里的Q,K,V全是Decoder自己的输入
        out = self.dec_enc_attn(out, memory, memory, enc_pad_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        return self.pos_ffn(out)  # [batch_size, tgt_len, d_model]


class TransformerDecoder(nn.Module):
    @staticmethod
    def get_attn_subsequence_mask(seq):
        """
        :param seq: [batch_size, tgt_len]
        """
        batch_size, tgt_len = seq.size(0), seq.size(1)
        attn_shape = (batch_size, tgt_len, tgt_len)
        subsequence_mask = triu(ones(attn_shape), diagonal=1).byte()
        return subsequence_mask.to(seq.device)  # [batch_size, tgt_len, tgt_len]

    def __init__(self, tgt_vocab_size, d_model, d_feed_forward, n_layers, n_heads, d_k, d_v, dropout, max_len):
        super(TransformerDecoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_feed_forward, n_heads, d_k, d_v) for _ in range(n_layers)])  # Decoder的blocks

    def forward(self, tgt, src, memory, dec_pad_mask, enc_pad_mask):
        """
        :param tgt: [batch_size, tgt_len]
        :param src: [batch_size, src_len]
        :param memory: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        :param dec_pad_mask: 
        :param enc_pad_mask: 
        """
        out = self.tgt_emb(tgt)  # [batch_size, tgt_len, d_model]
        out = self.pos_emb(out.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        attn_subsequence_mask = TransformerDecoder.get_attn_subsequence_mask(tgt)  # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        attn_mask = gt((dec_pad_mask + attn_subsequence_mask), 0)  # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

        for layer in self.layers:
            # out: [batch_size, tgt_len, d_model], attn: [batch_size, n_heads, tgt_len, tgt_len]
            # Decoder的Block是上一个Block的输出out（变化）和Encoder网络的输出memory（固定）
            out = layer(out, memory, attn_mask, enc_pad_mask)
        # out: [batch_size, tgt_len, d_model]
        return out


class TranslationTransformer(nn.Module):
    @staticmethod
    def get_attn_pad_mask(seq_q, seq_k):
        # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
        """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
        encoder和decoder都可能调用这个函数，所以seq_len视情况而定
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        """
        batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
        # [batch_size, 1, len_k], True is masked
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, d_feed_forward=1024, n_layers=4, n_heads=4, dropout=.1, max_len=5000):
        """
        :param src_vocab_size: 源词汇表大小
        :param tgt_vocab_size: 目标词汇表大小
        :param d_model: Embedding Size（token embedding和position编码的维度）
        :param d_feed_forward: FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
        :param n_layers: number of Encoder of Decoder Layer（Block的个数）
        :param n_heads: number of heads in Multi-Head Attention（有几套头）
        :param dropout: 细胞置零几率
        :param max_len: 位置编码长度
        """
        super(TranslationTransformer, self).__init__()
        self.src_vocab_size, self.tgt_vocab_size = src_vocab_size, tgt_vocab_size
        d_k = d_v = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
        self.encoder = TransformerEncoder(src_vocab_size, d_model, d_feed_forward, n_layers, n_heads, d_k, d_v, dropout, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, d_feed_forward, n_layers, n_heads, d_k, d_v, dropout, max_len)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, enc_pad_mask):
        """Transformers的输入：
        :param src: [batch_size, src_len]
        :param tgt: [batch_size, tgt_len]
        :param src_pad_mask: [batch_size, src_len, src_len]
        :param tgt_pad_mask: [batch_size, tgt_len, tgt_len]
        :param enc_pad_mask: [batch_size, tgt_len, src_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        memory = self.encoder(src, src_pad_mask)  # [batch_size, tgt_len, d_model]
        out = self.decoder(tgt, src, memory, tgt_pad_mask, enc_pad_mask)  # [batch_size, tgt_len, d_model]
        dec_logits = self.projection(out)  # [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))

    def load_state_dict(self, checkpoint):
        """加载权重，自动处理词汇表扩展"""
        old_weight = checkpoint['encoder.src_emb.weight']
        if old_weight.shape[0] != self.src_vocab_size:
            new_encoder_emb = randn(self.src_vocab_size, old_weight.shape[1], device=old_weight.device, dtype=old_weight.dtype)
            new_encoder_emb[:old_weight.shape[0]] = old_weight
            checkpoint['encoder.src_emb.weight'] = new_encoder_emb
        old_weight = checkpoint['decoder.tgt_emb.weight']
        if old_weight.shape[0] != self.tgt_vocab_size:
            new_decoder_emb = randn(self.tgt_vocab_size, old_weight.shape[1], device=old_weight.device, dtype=old_weight.dtype)
            new_decoder_emb[:old_weight.shape[0]] = old_weight
            checkpoint['decoder.tgt_emb.weight'] = new_decoder_emb

        old_weight = checkpoint['projection.weight']
        if old_weight.shape[0] != self.tgt_vocab_size:
            # 创建新的权重矩阵，保留旧权重，初始化新增部分
            new_proj_weight = randn(self.tgt_vocab_size, old_weight.shape[1], device=old_weight.device, dtype=old_weight.dtype)
            new_proj_weight[:old_weight.shape[0]] = old_weight
            checkpoint['projection.weight'] = new_proj_weight
        super().load_state_dict(checkpoint)
