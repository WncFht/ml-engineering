---
title: 下溢和上溢
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/n6vox1p4/
---
# 下溢和上溢检测

对于本节，我们将使用 [underflow_overflow](./underflow_overflow.py) 库。

如果您开始得到 `loss=NaN` 或者模型由于激活或权重中出现 `inf` 或 `nan` 而表现出其他异常行为，则需要发现第一次下溢或上溢发生在哪里以及导致它的原因。幸运的是，您可以通过激活一个特殊的模块来轻松完成此操作，该模块将自动进行检测。

让我们使用 `t5-large` 模型进行此演示。

```python
from .underflow_overflow import DebugUnderflowOverflow
from transformers import AutoModel

model = AutoModel.from_pretrained("t5-large")
debug_overflow = DebugUnderflowOverflow(model)
```

[`underflow_overflow.DebugUnderflowOverflow`] 将钩子插入到模型中，在每次
前向调用后立即测试输入和输出变量以及相应模块的权重。一旦在激活或权重的至少一个元素中检测到 `inf` 或
`nan`，程序将断言并打印如下报告（这是在 fp16 混合精度下用 `google/mt5-small` 捕获的）：

```
在 batch_number=0 期间检测到 inf/nan
最近 21 个前向帧：
abs min  abs max  元数据
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

为了简洁起见，示例输出在中间被修剪了。

第二列显示了绝对最大元素的值，所以如果您仔细查看最后几个帧，
输入和输出都在 `1e4` 的范围内。所以当这个训练在 fp16 混合精度下进行时，最后
一步溢出了（因为在 `fp16` 下，`inf` 前的最大数字是 `64e3`）。为了避免在
`fp16` 下溢出，激活值必须保持远低于 `1e4`，因为 `1e4 * 1e4 = 1e8`，所以任何与
大激活值的矩阵乘法都会导致数值溢出。

在跟踪的最初，您可以发现问题发生在哪个批次号（这里 `Detected inf/nan during batch_number=0` 意味着问题发生在第一个批次）。

每个报告的帧都以声明该帧报告的相应模块的完全限定条目开始
。如果我们只看这个帧：

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

在这里，`encoder.block.2.layer.1.layer_norm` 表示它是编码器第二个
块的第一层的层归一化。而 `forward` 的特定调用是 `T5LayerNorm`。

让我们看一下该报告的最后几帧：

```
在 batch_number=0 期间检测到 inf/nan
最近 21 个前向帧：
abs min  abs max  元数据
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

最后一帧报告了 `Dropout.forward` 函数，第一个条目是唯一的输入，第二个条目是
唯一的输出。您可以看到它是从 `DenseReluDense` 类中的 `dropout` 属性调用的。我们可以看到
它发生在第一个批次的第二个块的第一层。最后，绝对最大的
输入元素是 `6.27e+04`，而输出是 `inf`。

您可以在这里看到，`T5DenseGatedGeluDense.forward` 产生的输出激活，其绝对最大值
约为 62.7K，非常接近 fp16 的上限 64K。在下一帧中，我们有 `Dropout`，它在将某些元素归零后重新归一化
权重，这将绝对最大值推高到超过 64K，我们得到了一个
上溢 (`inf`)。

如您所见，当数字开始变得对于 fp16
数字来说非常大时，我们需要查看的是前面的帧。

让我们将报告与 [`models/t5/modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) 中的代码进行匹配：


```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

现在很容易看到 `dropout` 调用，以及之前的所有调用。

由于检测是在前向钩子中进行的，因此这些报告会在每次 `forward`
返回后立即打印。

回到完整的报告，为了处理它并解决问题，我们需要向上移动几帧，到数字
开始变大的地方，并且很可能在这里切换到 `fp32` 模式，这样数字在相乘
或相加时就不会溢出。当然，可能还有其他解决方案。例如，如果 `amp`
已启用，我们可以暂时关闭它，在将原始的 `forward` 移动到一个辅助包装器后，像这样：

```python
import torch

def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states

def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

由于自动检测器只报告完整帧的输入和输出，一旦你知道了要看哪里，你可能
也想分析任何特定 `forward` 函数的中间阶段。在这种情况下，你可以使用
`detect_overflow` 辅助函数在你想要的地方注入检测器，例如：

```python
from underflow_overflow import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

您可以看到我们添加了其中的 2 个，现在我们跟踪是否在中间的某个地方检测到 `forwarded_states` 的 `inf` 或 `nan`
。

实际上，检测器已经报告了这些，因为上面示例中的每个调用都是一个 `nn.Module`，但是
比方说，如果您有一些本地的直接计算，这就是您要做的。

此外，如果您在自己的代码中实例化调试器，您可以从
其默认值调整打印的帧数，例如：

```python
from .underflow_overflow import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

## 特定批次的绝对最小和最大值追踪

同样的调试类可以用于逐批次跟踪，同时关闭下溢/上溢检测功能。

假设您想观察给定批次的每个 `forward` 调用的所有成分的绝对最小值和最大值，并且只对批次 1 和 3 进行此操作。然后您可以这样实例化此类：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

现在，将使用与下溢/上溢检测器相同的格式来跟踪完整的批次 1 和 3。

批次是从 0 开始索引的。

如果您知道程序在某个批次号之后开始出现异常行为，这会很有帮助，因此您可以快速跳转到该区域。以下是此类配置的示例截断输出：

```
                  *** 开始批次号=1 ***
abs min  abs max  元数据
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     不是一个张量 output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     不是一个张量 output

                  *** 开始批次号=3 ***
abs min  abs max  元数据
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

在这里，您将获得大量的帧转储 - 与模型中的前向调用一样多，因此它可能
是您想要的，也可能不是，但有时它比普通的调试器更容易用于调试目的。例如，如果
问题在批次号 150 开始出现。因此您可以转储批次 149 和 150 的跟踪，并比较
数字开始发散的地方。

您还可以指定在哪个批次号之后停止训练，使用：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
