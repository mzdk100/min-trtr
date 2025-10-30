import os
from torch import nn, no_grad, LongTensor, ones, onnx

class GreedyDecoder(nn.Module):
    def __init__(self, decoder, projection):
        super().__init__()
        self.decoder = decoder
        self.projection = projection
        
    def forward(self, tgt, src, memory, tgt_pad_mask, enc_pad_mask):
        out = self.decoder(tgt, src, memory, tgt_pad_mask, enc_pad_mask)
        return self.projection(out)  # 通过全连接层生成词汇表的分布

def export_onnx(model, src_vocab, tgt_vocab, onnx_dir, example_src='hello world'):
    # 确保目录存在
    os.makedirs(onnx_dir, exist_ok=True)
    # 准备示例输入
    model.eval()
    encoder = model.encoder
    decoder = GreedyDecoder(model.decoder, model.projection)

    device = next(model.parameters()).device
    
    # 将示例句子转换为张量
    src_indexes = [src_vocab[word] for word in example_src.split()]
    src = LongTensor(src_indexes).unsqueeze(0).to(device)  # [1, seq_len]
    src_pad_mask = model.get_attn_pad_mask(src, src)  # [batch_size, src_len, src_len]
    memory = encoder(src, src_pad_mask)
    tgt = ones(1, 1).fill_(tgt_vocab['<sos>']).long().to(device)  # [1, 1]
    tgt_pad_mask = model.get_attn_pad_mask(tgt, tgt)  # [batch_size, tgt_len, tgt_len]
    enc_pad_mask = model.get_attn_pad_mask(tgt, src)  # [batc_size, tgt_len, src_len]

    # 导出ONNX模型
    with no_grad():
        onnx.export(
            encoder,
            args = (src, src_pad_mask),
            f=os.path.join(onnx_dir, 'translation_encoder.onnx'),
            input_names=['src', 'src_pad_mask'],
            output_names=['memory'],
            dynamic_axes={
              'src': {1: 'src_seq_len'},
              'src_pad_mask': {1: 'src_seq_len', 2: 'src_seq_len'},
            },
            dynamo=False,
            do_constant_folding=True,                         # whether to execute constant folding for optimization
            opset_version=20,
#            verbose=True
        )
        onnx.export(
            decoder,
            args = (tgt, src, memory, tgt_pad_mask, enc_pad_mask),
            f=os.path.join(onnx_dir, 'translation_decoder.onnx'),
            input_names=['tgt', 'src', 'memory', 'tgt_pad_mask', 'enc_pad_mask'],
            output_names=['output'],
            dynamic_axes={
              'tgt': {1: 'tgt_seq_len'},
              'src': {1: 'src_seq_len'},
              'memory': {1: 'memory_seq_len'},
              'tgt_pad_mask': {1: 'tgt_seq_len', 2: 'tgt_seq_len'},
              'enc_pad_mask': {1: 'tgt_seq_len', 2: 'src_seq_len'},
            },
            dynamo=False,
            do_constant_folding=True,                         # whether to execute constant folding for optimization
            opset_version=20,
#            verbose=True
        )
