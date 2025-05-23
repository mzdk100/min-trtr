# min-trtr

使用Transformer网络构建的机器翻译模型(Translation Model)，他注重轻量化，Transformer是重新实现的，并非torch中提供的标准模型，除了pytorch和jieba之外没有引用其他第三方python库，因此可以轻松构建和使用（包括从零训练和微调）。
在`infer`文件夹中还提供了一个rust的推理代码，方便跨平台部署。

## 配置环境

```shell
git clone https://github.com/mzdk100/min-trtr.git
cd min-trtr
python -m venv .venv
./.venv/Scripts/activate
pip install jieba
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 从零训练

1. 准备您的数据，请查看`data/dataset_train.txt`中的数据集格式，并编辑新增训练数据；
2. 给英文单词标注ID，请编辑`data/vocab_source.txt`新增源语言单词ID；
3. 给中文词语标注ID，请编辑`data/vocab_target.txt`新增目标语言单词ID；
4. 准备测试数据，查看`data/dataset_test.txt`；
5. 开始训练

```shell
python t3.py
```

6. 当训练完毕会自动保存模型到`checkpoint`文件夹中（包括onnx格式的模型）。

## 微调模型

您可以随时新增自己的数据到`data/dataset_train.txt`中，随时训练

```shell
python t3.py
```

命令和从零训练是一样的，但他会加载已有的模型，并会自动拓展词汇表（如果您新增了新单词），这不是从零训练，因此他将很快收敛。

## 验证模型

当训练完成后会输出关于模型的测试结果，你还可以使用下面的命令测试onnx格式的模型是否成功导出：

```shell
python onnxinf.py
```

如果没有任何报错则成功。

## Rust推理

1. 桌面平台

```shell
cd infer
cargo run --example infer
```

2. 安卓平台

```shell
cd infer/examples/android
cargo install apk2
./run
```

## 代码结构

- `t3.py`: 包含训练、测试和推理的主要逻辑。
- `dataset.py`: 数据集类和批处理函数。
- `model.py`: Transformer模型定义。
- `onnxexp.py`: ONNX模型导出函数。
- `infer/`: Rust推理代码目录。

## 数据格式

- `data/dataset_train.txt` 和 `data/dataset_test.txt`: 训练和测试数据集，每行格式为`源语言句子\t目标语言句子`。
- `data/vocab_source.txt` 和 `data/vocab_target.txt`: 词汇表文件，每行格式为`单词\tID`。

## 模型保存

训练完成后，模型会保存到`checkpoint/translation_model.pt`，同时也会导出ONNX格式的模型到`checkpoint`文件夹中。

## 注意事项

- 确保您的数据集和词汇表文件格式正确。
- 如果您在训练过程中遇到任何问题，请检查数据集和模型配置。
- Rust推理代码需要安装Rust和Cargo工具链，并确保`infer`目录中的依赖项已正确安装。

## 贡献

如果您有任何改进意见或想要贡献代码，请随时提交Pull Request或开Issue讨论。

## 许可证

本项目遵循MIT许可证。请查看`LICENSE`文件了解更多信息。