---
title: 测试
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/b1z8uix8/
---
# 编写和运行测试

注意：本文档的一部分引用了附带的 [testing_utils.py](testing_utils.py) 提供的功能，其中大部分是我在 HuggingFace 工作时开发的。

本文档涵盖了 `pytest` 和 `unittest` 的功能，并展示了如何将两者结合使用。


## 运行测试

### 运行所有测试

```console
pytest
```
我使用以下别名：
```bash
alias pyt="pytest --disable-warnings --instafail -rA"
```

这告诉 pytest：

- 禁用警告
- `--instafail` 在失败发生时立即显示，而不是在最后
- `-rA` 生成简短的测试摘要信息

这需要你安装：
```
pip install pytest-instafail
```


### 获取所有测试的列表

显示测试套件中的所有测试：

```bash
pytest --collect-only -q
```

显示给定测试文件中的所有测试：

```bash
pytest tests/test_optimization.py --collect-only -q
```

我使用以下别名：
```bash
alias pytc="pytest --disable-warnings --collect-only -q"
```

### 运行特定的测试模块

要运行单个测试模块：

```bash
pytest tests/utils/test_logging.py
```

### 运行特定的测试

如果使用 `unittest`，要运行特定的子测试，你需要知道包含这些测试的 `unittest` 类的名称。例如，可能是：

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

这里：

- `tests/test_optimization.py` - 包含测试的文件
- `OptimizationTest` - 测试类的名称
- `test_adam_w` - 特定测试函数的名称

如果文件包含多个类，你可以选择只运行给定类的测试。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest
```

将运行该类中的所有测试。

如前所述，你可以通过运行以下命令查看 `OptimizationTest` 类中包含哪些测试：

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

你可以通过关键字表达式来运行测试。

要只运行名称中包含 `adam` 的测试：

```bash
pytest -k adam tests/test_optimization.py
```

可以使用逻辑 `and` 和 `or` 来指示是应该匹配所有关键字还是匹配其中一个。`not` 可以用来取反。

要运行所有测试，除了名称中包含 `adam` 的测试：

```bash
pytest -k "not adam" tests/test_optimization.py
```

你也可以将两种模式结合在一个表达式中：

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如，要同时运行 `test_adafactor` 和 `test_adam_w`，你可以使用：

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

注意，我们在这里使用 `or`，因为我们希望匹配任一关键字以包含两者。

如果你只想包含同时包含两种模式的测试，则应使用 `and`：

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 只运行修改过的测试

您可以使用 [pytest-picked](https://github.com/anapaulagomes/pytest-picked) 来运行与未暂存文件或当前分支（根据 Git）相关的测试。这是一个快速测试您的更改没有破坏任何东西的好方法，因为它不会运行与您未触及的文件相关的测试。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

所有已修改但尚未提交的文件和文件夹中的测试都将被运行。

### 在源文件修改时自动重新运行失败的测试

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) 提供了一个非常有用的功能，即检测所有失败的测试，然后等待您修改文件并持续重新运行那些失败的测试，直到它们通过为止，而您则在修复它们。这样，您在修复后就无需重新启动 pytest。这个过程会一直重复，直到所有测试都通过，之后会再次执行一次完整的运行。

```bash
pip install pytest-xdist
```

要进入该模式：`pytest -f` 或 `pytest --looponfail`

文件更改是通过查看 `looponfailroots` 根目录及其所有内容（递归）来检测的。如果此值的默认设置不适合您，您可以在项目中通过在 `setup.cfg` 中设置一个配置选项来更改它：

```ini
[tool:pytest]
looponfailroots = transformers tests
```

或者 `pytest.ini`/`tox.ini` 文件：

```ini
[pytest]
looponfailroots = transformers tests
```

这将导致仅在相对于 ini 文件目录的指定目录中查找文件更改。

[pytest-watch](https://github.com/joeyespo/pytest-watch) 是此功能的另一个替代实现。


### 跳过一个测试模块

如果你想运行所有测试模块，除了少数几个，你可以通过给出一个明确的测试列表来排除它们。例如，要运行除了 `test_modeling_*.py` 之外的所有测试：

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

### 清除状态

CI 构建以及当隔离性很重要（相对于速度）时，应该清除缓存：

```bash
pytest --cache-clear tests
```

### 并行运行测试

如前所述，`make test` 通过 `pytest-xdist` 插件（`-n X` 参数，例如 `-n 2` 表示运行 2 个并行作业）并行运行测试。

`pytest-xdist` 的 `--dist=` 选项允许控制测试的分组方式。`--dist=loadfile` 将位于同一个文件中的测试放在同一个进程上。

由于执行测试的顺序不同且不可预测，如果使用 `pytest-xdist` 运行测试套件产生失败（意味着我们有一些未被检测到的耦合测试），请使用 [pytest-replay](https://github.com/ESSS/pytest-replay) 以相同的顺序重播测试，这应该有助于将失败的序列最小化。

### 测试顺序和重复

最好将测试重复几次，按顺序、随机或分组进行，以检测任何潜在的相互依赖和与状态相关的错误（teardown）。而直接的多次重复则有助于发现由深度学习的随机性暴露出来的一些问题。


#### 重复测试

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

然后多次运行每个测试（默认为 50 次）：

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

脚注：这个插件与 `pytest-xdist` 的 `-n` 标志不兼容。

脚注：还有另一个插件 `pytest-repeat`，但它不兼容 `unittest`。


#### 以随机顺序运行测试

```bash
pip install pytest-random-order
```

重要提示：`pytest-random-order` 的存在将自动随机化测试，无需更改配置或命令行选项。

如前所述，这允许检测耦合测试——即一个测试的状态影响另一个测试的状态。当安装了 `pytest-random-order` 时，它将打印该会话使用的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

因此，如果给定的特定序列失败，你可以通过添加该确切的种子来重现它，例如：

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

只有当你使用完全相同的测试列表（或根本没有列表）时，它才能重现确切的顺序。一旦你开始手动缩小列表范围，你就不能再依赖种子了，而必须按照它们失败的确切顺序列出它们，并告诉 pytest 不要随机化它们，而是使用 `--random-order-bucket=none`，例如：

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

要禁用所有测试的随机排序：

```bash
pytest --random-order-bucket=none
```

默认情况下，隐含了 `--random-order-bucket=module`，这将在模块级别上打乱文件。它还可以在 `class`、`package`、`global` 和 `none` 级别上打乱。有关完整详细信息，请参阅其[文档](https://github.com/jbasko/pytest-random-order)。

另一个随机化替代方案是：[`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly)。这个模块具有非常相似的功能/接口，但它没有 `pytest-random-order` 中可用的桶模式。它也存在一旦安装就会强制生效的问题。

### 外观和感觉的变化

#### pytest-sugar

[pytest-sugar](https://github.com/Frozenball/pytest-sugar) 是一个插件，可以改善外观和感觉，添加一个进度条，并立即显示失败的测试和断言。安装后会自动激活。

```bash
pip install pytest-sugar
```

要在没有它的情况下运行测试，请运行：

```bash
pytest -p no:sugar
```

或者卸载它。



#### 报告每个子测试的名称及其进度

对于单个或一组测试，通过 `pytest`（在 `pip install pytest-pspec` 之后）：

```bash
pytest --pspec tests/test_optimization.py
```

#### 立即显示失败的测试

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) 会立即显示失败和错误，而不是等到测试会话结束。

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### 使用 GPU 还是不使用 GPU

在启用 GPU 的设置中，要在仅 CPU 模式下进行测试，请添加 `CUDA_VISIBLE_DEVICES=""`：

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

或者如果你有多个 GPU，你可以指定 `pytest` 使用哪一个。例如，如果你有 GPU `0` 和 `1`，要只使用第二个 GPU，你可以运行：

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

当你想在不同的 GPU 上运行不同的任务时，这很方便。

有些测试必须在仅 CPU 上运行，其他测试可以在 CPU、GPU 或 TPU 上运行，还有一些测试必须在多个 GPU 上运行。以下 skip 装饰器用于设置测试的 CPU/GPU/TPU 要求：

- `require_torch` - 此测试将仅在 torch 下运行
- `require_torch_gpu` - 与 `require_torch` 相同，但还需要至少 1 个 GPU
- `require_torch_multi_gpu` - 与 `require_torch` 相同，但还需要至少 2 个 GPU
- `require_torch_non_multi_gpu` - 与 `require_torch` 相同，但还需要 0 或 1 个 GPU
- `require_torch_up_to_2_gpus` - 与 `require_torch` 相同，但还需要 0、1 或 2 个 GPU
- `require_torch_tpu` - 与 `require_torch` 相同，但还需要至少 1 个 TPU

让我们在下表中描述 GPU 的要求：


| GPU 数量 | 装饰器                         |
|--------|--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


例如，这是一个只有在有 2 个或更多 GPU 可用且安装了 pytorch 时才必须运行的测试：

```python no-style
from testing_utils import require_torch_multi_gpu

@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

这些装饰器可以堆叠使用：

```python no-style
from testing_utils import require_torch_gpu

@require_torch_gpu
@some_other_decorator
def test_example_slow_on_gpu():
```

像 `@parametrized` 这样的装饰器会重写测试名称，因此 `@require_*` 跳过装饰器必须放在最后才能正常工作。以下是正确用法的示例：

```python no-style
from testing_utils import require_torch_multi_gpu
from parameterized import parameterized

@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

这个顺序问题在 `@pytest.mark.parametrize` 中不存在，你可以把它放在最前面或最后面，它仍然可以工作。但它只适用于非 unittests。

在测试内部：

- 有多少个 GPU 可用：

```python
from testing_utils import get_gpu_count

n_gpu = get_gpu_count()
```


### 分布式训练

`pytest` 不能直接处理分布式训练。如果尝试这样做 - 子进程不会做正确的事情，最终会认为它们是 `pytest` 并开始循环运行测试套件。然而，如果一个正常的进程产生多个工作进程并管理 IO 管道，这是可行的。

以下是一些使用它的测试：

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/deepspeed/test_deepspeed.py)

要直接跳转到执行点，请在这些测试中搜索 `execute_subprocess_async` 调用，您将在 [testing_utils.py](testing_utils.py) 中找到它。

您将需要至少 2 个 GPU 才能看到这些测试的实际效果：

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

(`RUN_SLOW` 是 HF Transformers 使用的一个特殊装饰器，通常用于跳过重型测试)

### 输出捕获

在测试执行期间，发送到 `stdout` 和 `stderr` 的任何输出都会被捕获。如果一个测试或一个设置方法失败，其相应的捕获输出通常会与失败的回溯一起显示。

要禁用输出捕获并正常获取 `stdout` 和 `stderr`，请使用 `-s` 或 `--capture=no`：

```bash
pytest -s tests/utils/test_logging.py
```

要将测试结果发送到 JUnit 格式的输出：

```bash
py.test tests --junitxml=result.xml
```

### 颜色控制

要不使用颜色（例如，白色背景上的黄色是不可读的）：

```bash
pytest --color=no tests/utils/test_logging.py
```

### 将测试报告发送到在线 pastebin 服务

为每个测试失败创建一个 URL：

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

这将把测试运行信息提交到一个远程 Paste 服务，并为每个失败提供一个 URL。你可以像往常一样选择测试，或者添加例如 -x，如果你只想发送一个特定的失败。

为整个测试会话日志创建一个 URL：

```bash
pytest --pastebin=all tests/utils/test_logging.py
```








## 编写测试

大多数情况下，在同一个测试套件中结合使用 `pytest` 和 `unittest` 效果很好。你可以在[这里](https://docs.pytest.org/en/stable/unittest.html)阅读在这样做时支持哪些功能，但要记住的重要一点是，大多数 `pytest` 的 fixture 都不起作用。参数化也不起作用，但我们使用 `parameterized` 模块，它的工作方式类似。


### 参数化

通常，需要多次运行同一个测试，但使用不同的参数。这可以在测试内部完成，但这样就无法只用一组参数来运行该测试。

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert_equal(math.floor(input), expected)
```

现在，默认情况下，这个测试将运行 3 次，每次 `test_floor` 的最后 3 个参数将被赋予参数列表中的相应参数。

你可以只运行 `negative` 和 `integer` 两组参数：

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

或者除了 `negative` 子测试之外的所有测试：

```bash
pytest -k "not negative" tests/test_mytest.py
```

除了刚才提到的 `-k` 过滤器，你还可以找出每个子测试的确切名称，并使用它们的确切名称运行任意或所有子测试。

```bash
pytest test_this1.py --collect-only -q
```

它会列出：

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

所以现在你可以只运行两个特定的子测试：

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

[parameterized](https://pypi.org/project/parameterized/) 模块对 `unittests` 和 `pytest` 测试都有效。

然而，如果测试不是 `unittest`，你可以使用 `pytest.mark.parametrize`。

这是同一个例子，这次使用 `pytest` 的 `parametrize` 标记：

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert_equal(math.floor(input), expected)
```

与 `parameterized` 一样，使用 `pytest.mark.parametrize` 可以精确控制运行哪些子测试，如果 `-k` 过滤器不起作用的话。不过，这个参数化函数为子测试创建了一组略有不同的名称。它们看起来像这样：

```bash
pytest test_this2.py --collect-only -q
```

它会列出：

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

所以现在你可以只运行特定的测试：

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

和上一个例子一样。



### 文件和目录

在测试中，我们经常需要知道相对于当前测试文件的位置，这并不简单，因为测试可能从多个目录调用，或者可能位于不同深度的子目录中。辅助类 `testing_utils.TestCasePlus` 通过整理所有基本路径并提供简单的访问器来解决这个问题：

- `pathlib` 对象（全部完全解析）：

  - `test_file_path` - 当前测试文件路径，即 `__file__`
  - `test_file_dir` - 包含当前测试文件的目录
  - `tests_dir` - `tests` 测试套件的目录
  - `examples_dir` - `examples` 测试套件的目录
  - `repo_root_dir` - 仓库的根目录
  - `src_dir` - `src` 目录（即 `transformers` 子目录所在的位置）

- 字符串化路径——与上面相同，但这些返回的是字符串路径，而不是 `pathlib` 对象：

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

要开始使用这些，你只需要确保测试位于 `testing_utils.TestCasePlus` 的子类中。例如：

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

如果你不需要通过 `pathlib` 操作路径，或者你只需要一个字符串路径，你可以随时在 `pathlib` 对象上调用 `str()` 或使用以 `_str` 结尾的访问器。例如：

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

#### 临时文件和目录

使用唯一的临时文件和目录对于并行测试运行至关重要，这样测试就不会覆盖彼此的数据。我们还希望在每个创建它们的测试结束时删除临时文件和目录。因此，使用像 `tempfile` 这样的包来满足这些需求是至关重要的。

然而，在调试测试时，你需要能够看到临时文件或目录中发生了什么，并且你希望知道它的确切路径，而不是在每次测试重新运行时都随机化。

辅助类 `testing_utils.TestCasePlus` 最适合用于此类目的。它是 `unittest.TestCase` 的子类，所以我们可以很容易地在测试模块中继承它。

这是一个使用它的例子：

```python
from testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

此代码创建一个唯一的临时目录，并将 `tmp_dir` 设置为其位置。

- 创建一个唯一的临时目录：

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir` 将包含创建的临时目录的路径。它将在测试结束时自动删除。

- 创建一个我选择的临时目录，确保在测试开始前它是空的，并且在测试后不清空它。

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

这在调试时很有用，当你想要监控一个特定的目录并确保之前的测试没有在里面留下任何数据时。

- 你可以通过直接覆盖 `before` 和 `after` 参数来覆盖默认行为，从而导致以下行为之一：

  - `before=True`: 临时目录总是在测试开始时被清除。
  - `before=False`: 如果临时目录已经存在，任何现有文件将保留在那里。
  - `after=True`: 临时目录总是在测试结束时被删除。
  - `after=False`: 临时目录总是在测试结束时保持不变。


脚注：为了安全地运行相当于 `rm -r` 的命令，如果使用了显式的 `tmp_dir`，则只允许项目仓库检出的子目录，这样就不会错误地清除 `/tmp` 或文件系统的其他重要部分。即，请始终传递以 `./` 开头的路径。

脚注：每个测试可以注册多个临时目录，除非另有要求，否则它们都将被自动删除。


#### 临时覆盖 sys.path

如果你需要临时覆盖 `sys.path` 以便从另一个测试导入，例如，你可以使用 `ExtendSysPath` 上下文管理器。例子：


```python
import os
from testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 跳过测试

当发现一个 bug 并编写了一个新的测试，但 bug 尚未修复时，这很有用。为了能够将其提交到主仓库，我们需要确保在 `make test` 期间跳过它。

方法：

- **skip** 表示您期望您的测试仅在满足某些条件时才通过，否则 pytest 应该完全跳过运行该测试。常见的例子是在非 Windows 平台上跳过仅限 Windows 的测试，或者跳过依赖于当前不可用的外部资源（例如数据库）的测试。

- **xfail** 意味着你期望一个测试会因为某种原因而失败。一个常见的例子是为一个尚未实现的功能或一个尚未修复的 bug 编写的测试。当一个测试尽管被期望失败（用 `pytest.mark.xfail` 标记）却通过了，这就是一个 xpass，并将在测试摘要中报告。

两者之间的一个重要区别是，`skip` 不会运行测试，而 `xfail` 会。所以如果 buggy 的代码导致了一些会影响其他测试的坏状态，不要使用 `xfail`。

#### 实现

- 以下是如何无条件跳过整个测试：

```python no-style
@unittest.skip("这个 bug 需要修复")
def test_feature_x():
```

或者通过 pytest：

```python no-style
@pytest.mark.skip(reason="这个 bug 需要修复")
```

或者 `xfail` 的方式：

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

以下是如何根据测试内部的检查来跳过测试：

```python
def test_feature_x():
    if not has_something():
        pytest.skip("不支持的配置")
```

或者整个模块：

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag 缺失，跳过测试", allow_module_level=True)
```

或者 `xfail` 的方式：

```python
def test_feature_x():
    pytest.xfail("在 bug XYZ 修复之前预计会失败")
```

- 以下是如何在缺少某个导入时跳过模块中的所有测试：

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- 根据条件跳过测试：

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="需要 python3.6 或更高版本")
def test_feature_x():
```

或者：

```python no-style
@unittest.skipIf(torch_device == "cpu", "不能进行半精度测试")
def test_feature_x():
```

或者跳过整个模块：

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="不在 windows 上运行")
class TestClass():
    def test_feature_x(self):
```

更多细节、示例和方法请参见[这里](https://docs.pytest.org/en/latest/skipping.html)。



### 捕获输出

#### 捕获 stdout/stderr 输出

为了测试写入 `stdout` 和/或 `stderr` 的函数，测试可以使用 `pytest` 的 [capsys 系统](https://docs.pytest.org/en/latest/capture.html)访问这些流。以下是如何实现的：

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # 消费捕获的输出流
    # 可选：如果你想重放消费的流：
    sys.stdout.write(out)
    sys.stderr.write(err)
    # 测试：
    assert msg in out
    assert msg in err
```

当然，大多数时候，`stderr` 会作为异常的一部分出现，所以在这种情况下必须使用 try/except：

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "不是一个好值"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} 在异常中：\n{error}"
```

另一种捕获 stdout 的方法是使用 `contextlib.redirect_stdout`：

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # 可选：如果你想重放消费的流：
    sys.stdout.write(out)
    # 测试：
    assert msg in out
```

捕获 stdout 的一个重要潜在问题是它可能包含 `\r` 字符，在正常的 `print` 中会重置所有已打印的内容。`pytest` 没有问题，但使用 `pytest -s` 时，这些字符会包含在缓冲区中，所以为了能够让测试在有和没有 `-s` 的情况下都能运行，你必须对捕获的输出进行额外的清理，使用 `re.sub(r'~.*\r', '', buf, 0, re.M)`。

但是，我们有一个辅助的上下文管理器包装器来自动处理这一切，无论其中是否包含 `\r`，所以它很简单：

```python
from testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

这是一个完整的测试示例：

```python
from testing_utils import CaptureStdout

msg = "秘密消息\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"捕获到：{cs.out}，期望 {final}"
```

如果你想捕获 `stderr`，请改用 `CaptureStderr` 类：

```python
from testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

如果你需要同时捕获两个流，请使用父类 `CaptureStd`：

```python
from testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

另外，为了帮助调试测试问题，默认情况下，这些上下文管理器在退出上下文时会自动重放捕获的流。


#### 捕获 logger 流

如果你需要验证 logger 的输出，你可以使用 `CaptureLogger`：

```python
from transformers import logging
from testing_utils import CaptureLogger

msg = "测试 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

### 使用环境变量进行测试

如果你想测试特定测试中环境变量的影响，你可以使用一个辅助装饰器 `transformers.testing_utils.mockenv`

```python
from testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

有时需要调用外部程序，这需要在 `os.environ` 中设置 `PYTHONPATH` 以包含多个本地路径。辅助类 `testing_utils.TestCasePlus` 可以提供帮助：

```python
from testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # 现在调用外部程序，将 `env` 传递给它
```

根据测试文件是在 `tests` 测试套件下还是在 `examples` 下，它会正确地设置 `env[PYTHONPATH]` 以包含这两个目录中的一个，以及 `src` 目录，以确保测试是针对当前仓库进行的，最后还会包含在调用测试之前 `env[PYTHONPATH]` 中已经设置的任何内容。

这个辅助方法会创建 `os.environ` 对象的一个副本，所以原始对象保持不变。


### 获取可复现的结果

在某些情况下，你可能希望消除测试中的随机性。为了获得完全相同的可复现结果，你需要固定种子：

```python
seed = 42

# python RNG
import random

random.seed(seed)

# pytorch RNGs
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np

np.random.seed(seed)

# tf RNG
tf.random.set_seed(seed)
```

## 调试测试

要在警告发生的地方启动调试器，请执行以下操作：

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```


## 一个创建多个 pytest 报告的黑科技

这是我多年前做的一个大规模的 `pytest` 补丁，以帮助更好地理解 CI 报告。

要激活它，请添加到 `tests/conftest.py`（如果还没有，请创建它）：

```python
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

然后当你运行测试套件时，像这样添加 `--make-reports=mytests`：

```bash
pytest --make-reports=mytests tests
```

它将创建 8 个独立的报告：
```bash
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
```

所以现在，你不再只有一个包含所有内容的 `pytest` 输出，而是可以将每种类型的报告保存到各自的文件中。

这个功能在 CI 上最有用，它使得内省问题以及查看和下载单个报告都变得更加容易。

为不同的测试组使用不同的 `--make-reports=` 值，可以使每个组的报告分开保存，而不是相互覆盖。

所有这些功能都已经存在于 `pytest` 中，但是没有简单的方法来提取它，所以我添加了[testing_utils.py](testing_utils.py)中的猴子补丁覆盖。嗯，我确实问过是否可以将这个功能贡献给 `pytest`，但我的提议没有被接受。
