from torch import nn, optim, utils, isnan, LongTensor, load, save, full, long, zeros, bool, ones, cat
import jieba, os
from dataset import TranslationDataset, collate_fn
from model import TranslationTransformer
from onnxexp import export_onnx

def get_data_loader(src_vocab, tgt_vocab, train=True):
    source_sentences, target_sentences = TranslationDataset.get_raw_data(train=train)
    source_sentences = [line.split(' ') for line in source_sentences]
    target_sentences = [list(jieba.cut(line)) for line in target_sentences]
    dataset = TranslationDataset(source_sentences, target_sentences, src_vocab, tgt_vocab)
    return utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

def train(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss, i = 0, 0
        for src, tgt in data_loader:
            tgt_input = tgt[:, :-1]  # 包含SOS，不包含EOS
            tgt_output = tgt[:, 1:].reshape(-1)   # 不包含SOS，包含EOS，并将目标输出展平为1D张量
            src_pad_mask = TranslationTransformer.get_attn_pad_mask(src, src)  # [batch_size, src_len, src_len]
            # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
            tgt_pad_mask = TranslationTransformer.get_attn_pad_mask(tgt_input, tgt_input)  # [batch_size, tgt_len, tgt_len]
            # 这个mask主要用于encoder-decoder attention层
            # get_attn_pad_mask主要是src的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
            #                       tgt只是提供expand的size的
            enc_pad_mask = TranslationTransformer.get_attn_pad_mask(tgt_input, src)  # [batc_size, tgt_len, src_len]

            optimizer.zero_grad()
            output = model(src, tgt_input, src_pad_mask, tgt_pad_mask, enc_pad_mask)
            output = output.view(-1, output.shape[-1])
            loss = criterion(output, tgt_output)
            if isnan(loss): continue
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=.5)  # 对梯度进行裁剪以防止梯度爆炸
            optimizer.step()

            epoch_loss += loss.item()
            i += 1
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / i}')

def test(model, data_loader, tgt_vocab):
    id_count, correct_total = 0, 0
    eos_token = tgt_vocab['<eos>']
    
    for src, tgt in data_loader:
        batch_size, max_len = tgt.shape
        # 编码器处理
        src_attn_mask = TranslationTransformer.get_attn_pad_mask(src, src)  # [batch_size, src_len, src_len]
        memory = model.encoder(src, src_attn_mask)
        # 初始化解码器输入
        ys = full((batch_size, max_len), tgt_vocab['<pad>'], device=src.device, dtype=long)
        ys[:, 0] = tgt_vocab['<sos>']
        eos_reached = zeros(batch_size, dtype=bool, device=src.device)
        # 解码生成
        for i in range(max_len -1):
            if eos_reached.all():
                break

            active = ~eos_reached
            # 解码生成下一个token
            current_ys = ys[active, :i + 1]
            tgt_pad_mask = TranslationTransformer.get_attn_pad_mask(current_ys, current_ys)  # [batch_size, tgt_len, tgt_len]
            enc_pad_mask = TranslationTransformer.get_attn_pad_mask(current_ys, src[active])  # [batc_size, tgt_len, src_len]
            out = model.decoder(current_ys, src[active], memory[active], tgt_pad_mask, enc_pad_mask)
            projected = model.projection(out)
            next_tokens = projected[:, -1].argmax(dim=-1)

            # 更新生成的序列
            ys[active, i + 1] = next_tokens
            # 检查是否生成了EOS
            eos_reached[active] |= next_tokens == eos_token
        # 计算正确率
        correct_total += (ys == tgt).sum()
        id_count += batch_size * max_len
    
    accuracy = correct_total / id_count * 100 if id_count != 0 else 0
    print(f'Test accuracy: {accuracy:.2f}%')

def inference(model, src_sentence, src_vocab, tgt_vocab, max_len=500):
    map = {v: k for k, v in tgt_vocab.items()}

    model.eval()  # 设置模型为评估模式
    src_indexes = [src_vocab[word] for word in src_sentence.split()]
    src = LongTensor(src_indexes).unsqueeze(0).to(next(model.parameters()).device)
    src_pad_mask = TranslationTransformer.get_attn_pad_mask(src, src)  # [batch_size, src_len, src_len]
    memory = model.encoder(src, src_pad_mask)
    ys = ones(1, 1).fill_(tgt_vocab['<sos>']).long().to(src.device)

    for i in range(max_len):
        tgt_pad_mask = TranslationTransformer.get_attn_pad_mask(ys, ys)  # [batch_size, tgt_len, tgt_len]
        enc_pad_mask = TranslationTransformer.get_attn_pad_mask(ys, src)  # [batch_size, tgt_len, tgt_len]
        out = model.decoder(ys, src, memory, tgt_pad_mask, enc_pad_mask)
        projected = model.projection(out)

        prob = projected[:, -1, :].max(dim=-1)[1]  # 选择概率最大的词作为输出
        # 将生成的词拼接到解码器的输入中
        ys = cat([ys, prob.unsqueeze(0)], dim=1)
        if prob[0] == tgt_vocab['<eos>']:  # 如果生成了<end of sequence>标记，则停止生成
            break
        yield map[prob[0].item()]


if __name__ == "__main__":
    with open('data/vocab_source.txt', 'r', encoding='utf-8') as f1, open('data/vocab_target.txt', 'r', encoding='utf-8') as f2:                                                                                           
        src_vocab = {k: int(v) for k, v in [i.strip().split('\t') for i in f1]}                              
        tgt_vocab = {k: int(v) for k, v in [i.rstrip().split('\t') for i in f2]}                              

    # 模型初始化
    model_path = "checkpoint/translation_model.pt"
    model = TranslationTransformer(len(src_vocab), len(tgt_vocab))
    if os.path.exists(model_path):
        model.load_state_dict(load(model_path))

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=.1)  # 使用交叉熵损失函数，忽略填充标记的损失

    # 训练模型
    data_loader = get_data_loader(src_vocab, tgt_vocab)
    train(model, data_loader, optimizer, criterion, num_epochs=10)

    # 训练完成后保存模型
    save(model.state_dict(), model_path)

    # 测试模型
    data_loader = get_data_loader(src_vocab, tgt_vocab, train=False)
    test(model, data_loader, tgt_vocab)

    # 导出onnx
    export_onnx(model, src_vocab, tgt_vocab, 'checkpoint')

    # 流式推理
    while (msg := input('请输入英文句子:')) != 'exit':
        for word in inference(model, msg.lower(), src_vocab, tgt_vocab):
            print(word, end='')
        print()
