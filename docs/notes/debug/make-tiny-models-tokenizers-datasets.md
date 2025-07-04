---
title: make-tiny-models-tokenizers-datasets
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/6cbi6zvt/
---
# 使用微型模型、分词器和数据集加快调试和开发速度

如果您使用全尺寸模型和分词器进行调试和开发，您的工作效率可能不高。不仅解决问题要困难得多，而且等待程序重新启动并到达所需点的时间也可能很长 - 从累积的角度来看，这可能会极大地消耗一个人的积极性和生产力，更不用说如果可能的话，解决问题需要更长的时间。

解决方案很简单：

**除非您正在测试模型的质量，否则请始终使用带有潜在微型分词器的微型随机模型。**

此外，大型模型通常需要大量资源，这些资源通常很昂贵，并且也可能使调试过程变得非常复杂。例如，任何调试器都可以处理单个进程，但如果您的模型不适合并需要某种需要多个进程的[并行化](../training/model-parallelism)，大多数调试器要么会崩溃，要么无法为您提供所需的功能。理想的开发环境是单个进程和一个保证可以容纳在最便宜的单个最小消费级 GPU 上的微型模型。如果您身边没有 GPU，您甚至可以使用免费的 [Google Colab](https://colab.research.google.com/) 来进行紧急开发。

因此，更新后的 ML 开发口头禅变成了：

- 模型越大，最终产品的生成效果越好
- 模型越小，最终产品的训练开始得越快

脚注：最近的研究表明，越大并不总是越好，但这足以传达我的沟通的重要性。

一旦您的代码正常工作，请切换到真实模型以测试您的生成质量。但即使在这种情况下，也请先尝试能产生高质量结果的最小模型。只有当您能看到生成基本正确时，才使用最大的模型来验证您的工作是否完美。

## 制作一个微型模型

重要提示：鉴于它们的受欢迎程度和设计良好的简单 API，我将讨论 HF [`transformers`](https://github.com/huggingface/transformers/) 模型。但同样的原则可以应用于任何其他模型。

简而言之：制作一个微型 HF `transformers` 模型很简单：

1. 获取全尺寸模型的配置对象
2. 缩小隐藏大小以及可能有助于模型体积的其他一些参数
3. 从缩小的配置中创建模型
4. 保存此模型。完成！

脚注：重要的是要记住，这将生成一个随机模型，所以不要期望其输出有任何质量。

脚注：这些笔记是针对 HF Transformers 模型编写的。如果您使用的是不同的建模库，您可能需要调整其中一些内容。

现在让我们来看一下实际代码，并将 ["google/mt5-small"](https://huggingface.co/google/mt5-small/tree/main) 转换为其微型随机对应物。

```
from transformers import MT5Config, MT5ForConditionalGeneration

mname_from = "google/mt5-small"
mname_very_small = "mt5-tiny-random"

config = MT5Config.from_pretrained(mname_from)

config.update(dict(
    d_model=64,
    d_ff=256,
))
print("new config", config)

very_small_model = MT5ForConditionalGeneration(config)
print(f"num of params {very_small_model.num_parameters()}")

very_small_model.save_pretrained(mname_very_small)
```

如您所见，这很容易做到。如果您不需要隐藏大小至少为 64，您可以使其更小。例如，尝试 8 - 您只需要确保注意力头的数量不大于隐藏大小即可。

另外请注意，您不需要任何 GPU 来执行此操作，您甚至可以在像 [BLOOM-176B](https://huggingface.co/bigscience/bloom) 这样的 176B 参数的巨大模型上执行此操作。因为您从不加载实际的原始模型，只加载其配置对象。

在修改配置之前，您可以转储原始参数并选择缩小更多维度。例如，使用更少的层可以使其更小，更容易调试。所以您可以这样做：

```
config.update(dict(
    d_model=64,
    d_ff=256,
    d_kv=8,
    num_layers=8,
    num_decoder_layers=8,
    num_heads=4,
    relative_attention_num_buckets=32,
))
```

原始的 ["google/mt5-small"](https://huggingface.co/google/mt5-small/tree/main) 模型文件为 1.2GB。通过上述更改（以及后续章节中解释的词汇表缩小），我们将其缩小到 126MB。

如果您正在处理多级嵌套配置，则需要分别更新每个子级的配置对象。例如，在 [IDEFICS](https://huggingface.co/HuggingFaceM4/idefics-9b/blob/main/config.json) 中，我们有 1 个主对象和 2 个嵌套对象：
```
config
config.perceiver_config
config.vision_config
```
如果您想缩小此模型，您需要使用较小的值更新 `config` 和 `config.vision_config`：
```
config.update(dict(
    hidden_size=64,
    intermediate_size=37,
    num_hidden_layers=5,
    num_attention_heads=4,
    max_position_embeddings=64,
    max_sequence_length=64,

))
# 子对象需要直接更新
config.vision_config.update(dict(embed_dim=64))
```
有关完整的可用脚本，请参阅 [idefics-make-tiny-model.py](tiny-scripts/idefics-make-tiny-model.py)（我没有费心添加词汇表缩小，因为我只是在这里演示如何更新嵌套配置对象）。

然后我们可以通过在保存前将模型转换为 fp16 或 bf16（取决于目标）来进一步将我们的微型模型大小减半：

```
very_small_model.half() # 转换为 fp16
#very_small_model.bfloat16() # 转换为 bf16
very_small_model.save_pretrained(mname_very_small)
```
这使我们得到了一个 64M 的文件。

所以你可以在这里停下来，你的程序已经可以更快地启动了。

还有一步你可以做，让它真正变得微小。

到目前为止，我们还没有缩小词汇维度，所以 64x250k (hidden*vocab) 仍然是巨大的。当然，这个 250k 词汇模型不是典型的 - 通常模型的词汇在 30-50k 左右，但即使是 30k，如果我们想要模型真正微小，也是很大的。

所以接下来我们将研究各种缩小分词器的技术，因为它定义了我们的词汇大小。

## 制作一个微小的分词器

这个任务的难易程度从一个相对简单的过程到一个更复杂的锻炼，这取决于底层分词器。

以下食谱来自 Hugging Face 的几位了不起的分词器专家，然后我根据我的需要对它们进行了调整。

在您真正需要它们之前，您可能并不需要真正理解这些是如何工作的，因此，如果您是第一次阅读本文，可以安全地跳过这些内容，直接进入[使用微型分词器制作微型模型](#making-a-tiny-model-with-a-tiny-tokenizer)。

### Anthony Moi 的版本

[Anthony Moi](https://github.com/n1t0) 的分词器缩减器：

```
import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer

vocab_keep_items = 5000
mname = "microsoft/deberta-base"

tokenizer = AutoTokenizer.from_pretrained(mname, use_fast=True)
assert tokenizer.is_fast, "This only works for fast tokenizers."
tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
vocab = tokenizer_json["model"]["vocab"]
if tokenizer_json["model"]["type"] == "BPE":
    new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
    merges = tokenizer_json["model"]["merges"]
    new_merges = []
    for i in range(len(merges)):
        a, b = merges[i].split()
        new_token = "".join((a, b))
        if a in new_vocab and b in new_vocab and new_token in new_vocab:
            new_merges.append(merges[i])
    tokenizer_json["model"]["merges"] = new_merges
elif tokenizer_json["model"]["type"] == "Unigram":
    new_vocab = vocab[:vocab_keep_items]
elif tokenizer_json["model"]["type"] == "WordPiece" or tokenizer_json["model"]["type"] == "WordLevel":
    new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
else:
    raise ValueError(f"don't know how to handle {tokenizer_json['model']['type']}")
tokenizer_json["model"]["vocab"] = new_vocab
tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
tokenizer.save_pretrained(".")
```

后来我发现 gpt2 似乎在词汇表的最后藏了一个特殊标记 `"<|endoftext|>"`，所以它被丢弃了，代码也坏了。所以我用以下代码把它加了回来：
```
if "gpt2" in mname:
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items-1 }
        new_vocab["<|endoftext|>"] = vocab_keep_items-1
    else:
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
```


### Lysandre Debut 的版本

[Lysandre Debut](https://github.com/LysandreJik) 使用 `train_new_from_iterator` 的缩减器：

```
from transformers import AutoTokenizer

mname = "microsoft/deberta-base" # or any checkpoint that has a fast tokenizer.
vocab_keep_items = 5000

tokenizer = AutoTokenizer.from_pretrained(mname)
assert tokenizer.is_fast, "This only works for fast tokenizers."
tokenizer.save_pretrained("big-tokenizer")
# Should be a generator of list of texts.
training_corpus = [
    ["This is the first sentence.", "This is the second one."],
    ["This sentence (contains #) over symbols and numbers 12 3.", "But not this one."],
]
new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_keep_items)
new_tokenizer.save_pretrained("small-tokenizer")
```
但这个需要一个训练语料库，所以我想到了一个作弊的方法，用它自己的原始词汇来训练新的分词器，结果是：

```
from transformers import AutoTokenizer

mname = "microsoft/deberta-base"
vocab_keep_items = 5000

tokenizer = AutoTokenizer.from_pretrained(mname)
assert tokenizer.is_fast, "This only works for fast tokenizers."
vocab = tokenizer.get_vocab()
training_corpus = [ vocab.keys() ] # Should be a generator of list of texts.
new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_keep_items)
new_tokenizer.save_pretrained("small-tokenizer")
```

这几乎是完美的，只是它现在没有任何关于每个单词/字符频率的信息（大多数分词器都是这样计算它们的词汇表的），如果你需要这些信息，你可以通过
让每个键出现 `len(vocab) - ID` 次来解决：

```
training_corpus = [ (k for i in range(vocab_len-v)) for k,v in vocab.items() ]
```
这将使脚本完成的时间长得多。

但对于一个微型模型（测试）的需求来说，频率根本不重要。



### 破解分词器文件的方法

有些分词器可以直接在文件级别进行截断，例如，让我们将 Llama2 的分词器缩小到 3k 项：

```
# 缩小原始词汇表以保持小巧（刚好能对任何单词进行分词，所以字母+符号）
# ElectraTokenizerFast 完全由一个 tokenizer.json 定义，其中包含词汇表和 id，
# 所以我们只需要明智地截断它
import subprocess
import shlex
from transformers import LlamaTokenizerFast

mname = "meta-llama/Llama-2-7b-hf"
vocab_keep_items = 3000

tokenizer_fast = LlamaTokenizerFast.from_pretrained(mname)
tmp_dir = f"/tmp/{mname}"
tokenizer_fast.save_pretrained(tmp_dir)
# 调整 tokenizer.json（vocab.txt 将在 save_pretrained 时自动调整大小）
# perl  -0777 -pi -e 's|(2999).*|$1},"merges": []}}|msg' tokenizer.json # 0-indexed, so vocab_keep_items-1!
closing_pat = '},"merges": []}}'
cmd = (f"perl -0777 -pi -e 's|({vocab_keep_items-1}).*|$1{closing_pat}|msg' {tmp_dir}/tokenizer.json")
#print(f"Running:\n{cmd}")
result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
# 使用修改后的分词器重新加载
tokenizer_fast_tiny = LlamaTokenizerFast.from_pretrained(tmp_dir)
tokenizer_fast_tiny.save_pretrained(".")
```
请记住，结果仅对功能测试有用 - 不适用于质量工作。

这是 [make_tiny_model.py](https://huggingface.co/stas/tiny-random-llama-2/blob/main/make_tiny_model.py) 的完整版本，其中既包括模型又包括分词器的缩小。


### SentencePiece 词汇表缩小

首先将 SentencePiece 克隆到父目录中：
```
git clone https://github.com/google/sentencepiece
```
现在进行缩小：
```
# 变通方法解决快速分词器 protobuf 问题，而且速度也快得多！
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import XLMRobertaTokenizerFast

mname = "xlm-roberta-base"

# 缩小原始词汇表以保持小巧
vocab_keep_items = 5000
tmp_dir = f"/tmp/{mname}"
vocab_orig_path = f"{tmp_dir}/sentencepiece.bpe.model" # 这个名字可能不同
vocab_short_path = f"{tmp_dir}/spiece-short.model"
# HACK: 需要 sentencepiece 源码来获取 sentencepiece_model_pb2，因为它没有被安装
sys.path.append("../sentencepiece/python/src/sentencepiece")
import sentencepiece_model_pb2 as model
tokenizer_orig = XLMRobertaTokenizerFast.from_pretrained(mname)
tokenizer_orig.save_pretrained(tmp_dir)
with open(vocab_orig_path, 'rb') as f: data = f.read()
# 改编自 https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/
m = model.ModelProto()
m.ParseFromString(data)
print(f"从原始的 {len(m.pieces)} 个词典项中缩小词汇表")
for i in range(len(m.pieces) - vocab_keep_items): _ = m.pieces.pop()
print(f"新词典 {len(m.pieces)}")
with open(vocab_short_path, 'wb') as f: f.write(m.SerializeToString())
m = None

tokenizer_fast_tiny = XLMRobertaTokenizerFast(vocab_file=vocab_short_path)
tokenizer_fast_tiny.save_pretrained(".")
```


## 用微小的分词器制作一个微小的模型

所以现在你可以将词汇表大小缩小到分词器允许的最小程度，也就是说，你至少需要有足够的标记来覆盖目标字母表和特殊字符，通常 3-5k 个标记就足够了。有时你甚至可以做得更小，毕竟原始的 ASCII 字符集只有 128 个字符。

如果我们继续本章前面的 MT5 代码，并添加前几节的分词器缩小代码，我们最终会得到这个脚本 [mt5-make-tiny-model.py](https://huggingface.co/stas/mt5-tiny-random/blob/main/mt5-make-tiny-model.py)，当我们运行它时——我们最终的模型文件非常小——只有 3.34 MB！正如您所看到的，该脚本还包含验证模型是否可以实际使用修改后的分词器的代码。结果将是垃圾，但目的是测试新模型和分词器是否功能正常。

这是另一个例子 [fsmt-make-super-tiny-model.py](https://huggingface.co/stas/tiny-wmt19-en-ru/blob/main/fsmt-make-super-tiny-model.py) - 在这里你可以看到我正在从头开始创建一个全新的微小词汇表。

我也建议总是将构建脚本与模型一起存储，以便您可以快速修复问题或制作类似版本的模型。

另外请注意，由于 HF `transformers` 需要微型模型进行测试，您很可能已经可以为每个架构找到一个，主要来自
https://huggingface.co/hf-internal-testing（只是他们没有包含如何制作它们的代码，但您现在可以根据这些笔记弄清楚）。

另一个提示：如果您需要一个稍微不同的微型模型，您也可以从一个已经存在的微型模型开始并进行调整。由于它是随机的，所以实际上只关乎获得正确的维度。例如，如果您找到的微型模型有 2 层，但您需要 8 层，只需用这个更大的维度重新保存它就可以了。




## 制作一个微小的数据集

与模型和分词器类似，拥有一个方便的微型版本的数据集，您经常使用它会很有帮助。像往常一样，这无助于质量测试，但它非常适合快速启动您的程序。

脚注：如果您使用的是已经预先索引的 Arrow 文件数据集，那么使用微型数据集的影响不会像使用微型模型那样巨大，因为这些数据集已经非常快了。但是，比如说，您希望迭代器在 10 个步骤内完成一个 epoch。与其编辑您的代码来截断数据集，不如直接使用一个微型数据集。

制作微型数据集的过程有点难以解释，因为它取决于原始模型的构建者，而这些构建者可能彼此大相径庭，但也许您可以将我的方法与您的数据集联系起来。

但这个概念仍然非常简单：

1. 克隆完整的数据集 git 仓库
2. 将其完整的数据 tarball 替换为一个只包含几个样本的微型 tarball
3. 保存它 - 完成！

以下是一些示例：

- [stas/oscar-en-10k](https://huggingface.co/datasets/stas/oscar-en-10k/blob/main/oscar-en-10k.py)
- [stas/c4-en-10k](https://huggingface.co/datasets/stas/c4-en-10k/blob/main/c4-en-10k.py)
- [stas/openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k/blob/main/openwebtext-10k.py)

在所有这些中，我拿了原始的 tarball，抓取了前 10k 条记录，重新打包，使用了这个更小的 tarball，就这样。构建脚本的其余部分基本保持不变。

这里有一些合成数据集的例子，我不是简单地缩小原始的 tarball，而是解压它，手动选择有代表性的例子，然后写一个脚本来根据这些少数有代表性的样本构建任何大小的所需数据集：
- [stas/general-pmd-synthetic-testing](https://huggingface.co/datasets/stas/general-pmd-synthetic-testing/blob/main/general-pmd-synthetic-testing.py) 和 [解包器](https://huggingface.co/datasets/stas/general-pmd-synthetic-testing/blob/main/general-pmd-ds-unpack.py)
- [stas/cm4-synthetic-testing](https://huggingface.co/datasets/stas/cm4-synthetic-testing/blob/main/cm4-synthetic-testing.py) - 和 [解包器](https://huggingface.co/datasets/stas/cm4-synthetic-testing/blob/main/m4-ds-unpack.py)

这些也是复杂的例子，其中每个样本不仅仅是一个文本条目，还可能有多个文本条目和图像。

解包器是将每个复杂的多记录样本扩展到其自己的子目录中，这样您现在就可以轻松地根据自己的喜好进行调整。您可以添加图像、删除它们、使文本记录更小等。您还会注意到我正在将大图像缩小为微小的 32x32 图像，所以再次，我正在应用在所有不破坏目标代码库要求的维度上使用微小的重要原则。

然后主脚本使用该结构来构建一个任何所需长度的数据集。

例如，这里是为 [stas/general-pmd-synthetic-testing](https://huggingface.co/datasets/stas/general-pmd-synthetic-testing/) 部署这些脚本的说明：

```
# 准备数据集仓库
https://huggingface.co/new-dataset => stas/general-pmd-synthetic-testing
git clone https://huggingface.co/datasets/stas/general-pmd-synthetic-testing
cd general-pmd-synthetic-testing

# 选择一些种子记录，以便有一些较长和较短的文本，有图像和没有图像的记录，
# 每种类型的几个变体
rm -rf data
python general-pmd-ds-unpack.py --dataset_name_or_path \
general_pmd/image/localized_narratives__ADE20k/train/00000-00002 --ids 1-10 --target_path data

cd data

# 缩小到最大 32x32，保持比例
mogrify -format jpg -resize 32x32\> */*jpg

# 将一条记录调整为没有图像和文本
cd 1
rm image.jpg text.txt
touch image.null text.null
cd -

cd ..

# 创建 tarball
tar -cvzf data.tar.gz data

# 完成数据集仓库
echo "该数据集旨在用于测试。它源自 general-pmd/localized_narratives__ADE20k \
数据集" >> README.md

# 测试数据集
cd ..
datasets-cli test general-pmd-synthetic-testing/general-pmd-synthetic-testing.py --all_configs
```

我也建议总是将构建脚本与数据集一起存储，以便您可以快速修复问题或制作类似版本的数据集。

与微型模型类似，您会在 https://huggingface.co/hf-internal-testing 下找到许多微型数据集。


## 结论

虽然在机器学习领域，我们有数据集、模型和分词器——每一个都可以做得很小，从而以低资源需求实现超高速开发，但如果您来自不同的行业，您可以将本章中讨论的思想应用到您特定领域的工件/有效载荷中。


## 本章中所有脚本的备份

如果您在阅读本章时，本章所指向的原始脚本消失或 HF 中心宕机，这里是[它们所有脚本的本地备份](./tiny-scripts/)。

自我提醒：要制作本章中链接到的文件的最新备份，请运行：
```
perl -lne 'while (/(https.*?.py)\)/g) { $x=$1; $x=~s/blob/raw/; print qq[wget $x] }' make-tiny-models.md
```
