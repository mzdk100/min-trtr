import torch, onnxruntime


src = torch.tensor([[3]], dtype=torch.long)
src_pad_mask = torch.zeros(1, 1, 1, dtype=torch.bool)
tgt = torch.tensor([[1]], dtype=torch.long)
tgt_pad_mask = torch.zeros(1, 1, 1, dtype=torch.bool)
enc_pad_mask = torch.zeros(1, 1, 1, dtype=torch.bool)



encoder = onnxruntime.InferenceSession('checkpoint/translation_encoder.onnx')
memory = encoder.run(None, {
  'src': src.numpy(),
  'src_pad_mask': src_pad_mask.numpy()
})[0]
decoder = onnxruntime.InferenceSession('checkpoint/translation_decoder.onnx')
outputs = decoder.run(None, {
    'tgt': tgt.numpy(),
    'memory': memory,
    'tgt_pad_mask': tgt_pad_mask.numpy(),
    'enc_pad_mask': enc_pad_mask.numpy(),
})
prob = outputs[0][0].argmax()
print(prob)