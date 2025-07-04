---
title: 工具
createTime: 2025/07/03 00:05:24
permalink: /notes/notes/7b8u3irq/
---
# 调试工具

## git 相关工具


### 有用的别名

显示当前分支中所有已修改文件与 HEAD 的差异：
```
alias brdiff="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff origin/\$def_branch..."
```

同上，但忽略空白差异，添加 `--ignore-space-at-eol` 或 `-w`：
```
alias brdiff-nows="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff -w origin/\$def_branch..."
```

列出当前分支中与 HEAD 相比所有已添加或修改的文件：
```
alias brfiles="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff --name-only origin/\$def_branch..."
```

一旦我们有了列表，我们现在可以自动打开一个编辑器来只加载已添加和修改的文件：
```
alias bremacs="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); emacs \$(git diff --name-only origin/\$def_branch...) &"
```


### git-bisect

（自我提醒：这是从 `the-art-of-debugging/methodology.md` 同步过来的，那里是真正的来源）

接下来讨论的方法应该适用于任何支持二分法的版本控制系统。我们将在本次讨论中使用 `git bisect`。

`git bisect` 有助于快速找到导致某个问题的提交。

用例：假设您正在使用 `transformers==4.33.0`，然后您需要一个更新的功能，所以您升级到了最前沿的 `transformers@main`，然后您的代码坏了。在这两个版本之间可能有数百个提交，通过遍历所有提交来找到导致问题的正确提交会非常困难。以下是您如何快速找出哪个提交是罪魁祸首的方法。

脚注：HuggingFace Transformers 在不经常破坏方面实际上做得很好，但鉴于其复杂性和庞大的规模，这种情况仍然会发生，一旦报告，问题会很快得到解决。由于它是一个非常流行的机器学习库，因此它是一个很好的调试用例。

解决方案：在已知的良好提交和不良提交之间对所有提交进行二分查找，以找到应该负责的那个提交。

我们将使用 2 个 shell 终端：A 和 B。终端 A 将用于 `git bisect`，终端 B 将用于测试您的软件。没有技术上的理由您不能只使用一个终端，但使用 2 个会更容易。

1. 在终端 A 中，获取 git 仓库并以开发模式（`pip install -e .`）将其安装到您的 Python 环境中。
```
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
现在，当您运行您的应用程序时，将自动使用此克隆的代码，而不是您之前从 PyPi 或 Conda 或其他地方安装的版本。

为了简单起见，我们还假设所有依赖项都已安装。

2. 接下来我们启动二分查找 - 在终端 A 中，运行：

```
git bisect start
```

3. 发现最后一个已知的良好提交和第一个已知的不良提交

`git bisect` 只需要 2 个数据点就可以工作。它需要知道一个已知可以工作的早期提交 (`good`) 和一个已知会破坏的后期提交 (`bad`)。所以如果你看一个给定分支上的提交序列，它会有 2 个已知点和许多围绕这些点的未知质量的提交：

```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->---------------->----------------> time
```

因此，例如，如果您知道 `transformers==4.33.0` 是好的，而 `transformers@main` (`HEAD`) 是坏的，请访问[发布页面](https://github.com/huggingface/transformers/releases)并搜索 `4.33.0`，找出与标签 `4.33.0` 对应的提交。我们发现它是带有 SHA [`5a4f340d`](https://github.com/huggingface/transformers/commit/5a4f340df74b42b594aedf60199eea95cdb9bed0) 的提交。

脚注：通常前 8 个十六进制字符足以作为给定仓库的唯一标识符，但您可以使用完整的 40 个字符的字符串。


所以现在我们指定哪个是第一个已知的良好提交：
```
git bisect good 5a4f340d
```

正如我们所说，我们将使用 `HEAD`（最新提交）作为坏的提交，在这种情况下，我们可以使用 `HEAD` 而不是找出相应的 SHA 字符串：
```
git bisect bad HEAD
```

但是，如果您知道它在 `4.34.0` 中坏了，您可以如上所述找到它的最新提交并使用它而不是 `HEAD`。

我们现在已经准备好找出是哪个提交给您带来了麻烦。

在你告诉 `git bisect` 好和坏的提交之后，它已经切换到了中间的某个提交：

```
...... orig_good ..... .... current .... .... ..... orig_bad ........
------------->--------------->---------------->----------------> time
```

您可以运行 `git log` 来查看它切换到了哪个提交。

提醒一下，我们以 `pip install -e .` 的方式安装了这个仓库，所以 Python 环境会立即更新到当前提交的代码版本。

4. 好还是坏

下一阶段是告诉 `git bisect` 当前提交是 `good` 还是 `bad`：

为此，在终端 B 中运行一次您的程序。

然后在终端 A 中运行：
```
git bisect bad
```
如果失败，或者：
```
git bisect good
```
如果成功。


例如，如果结果是坏的，`git bisect` 会在内部将最后一个提交标记为新的坏提交，并再次将提交减半，切换到一个新的当前提交：
```
...... orig_good ..... current .... new_bad .... ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

反之，如果结果是好的，那么你将得到：
```
...... orig_good ..... .... new_good .... current ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

5. 重复直到没有剩余的提交

继续重复第 4 步，直到找到有问题的提交。

完成二分查找后，`git bisect` 会告诉您是哪个提交导致了问题。

```
...... orig_good ..... .... last_good first_bad .... .. orig_bad ....
------------->--------------->---------------->----------------> time
```
如果您遵循了小的提交图，它将对应于 `first_bad` 提交。

然后您可以转到 `https://github.com/huggingface/transformers/commit/` 并在该 url 后附加提交 SHA，这将带您到该提交（例如 `https://github.com/huggingface/transformers/commit/57f44dc4288a3521bd700405ad41e90a4687abc0`），然后它将链接到它起源的 PR。然后您可以通过在该 PR 中跟进寻求帮助。

如果您的程序运行时间不长，即使有数千个提交要搜索，您也面临着 `2**n` 的 `n` 个二分步骤，因此 1024 个提交可以在 10 个步骤中搜索完。

如果您的程序非常慢，请尝试将其简化为一些小的东西 - 理想情况下是一个能够快速显示问题的小型重现程序。通常，注释掉您认为与当前问题无关的大块代码就足够了。

如果您想查看进度，可以要求它显示当前要检查的剩余提交范围：
```
git bisect visualize --oneline
```

6. 清理

所以现在将 git 仓库克隆恢复到您开始时的相同状态（很可能是 `HEAD`）：
```
git bisect reset
```

在向维护人员报告问题时，可能会重新安装库的良好版本。

有时，问题源于有意的向后不兼容的 API 更改，您可能只需要阅读项目的文档以查看发生了什么变化。例如，如果您从 `transformers==2.0.0` 切换到 `transformers==3.0.0`，几乎可以肯定您的代码会中断，因为主版本号的差异通常用于引入重大的 API 更改。


7. 可能的问题及其解决方案：

a. 跳过

如果由于某种原因无法测试当前提交 - 可以使用以下命令跳过：
```
git bisect skip
```
然后 `git bisect` 将继续对剩余的提交进行二分查找。

如果某个 API 在提交范围中间发生了变化，并且您的程序开始因为完全不同的原因而失败，这通常很有帮助。

您也可以尝试制作一个适应新 API 的程序变体，并改用它，但这并不总是那么容易。

b. 颠倒顺序

通常 git 期望 `bad` 在 `good` 之后。


```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

现在，如果 `bad` 在 `good` 修订版之前，并且您想找到第一个修复了先前存在的问题的修订版 - 您可以颠倒 `good` 和 `bad` 的定义 - 使用重载的逻辑状态会令人困惑，因此建议改用一组新的状态 - 例如，`fixed` 和 `broken` - 以下是操作方法。

```
git bisect start --term-new=fixed --term-old=broken
git bisect fixed
git bisect broken 6c94774
```
然后使用：
```
git fixed / git broken
```
代替：
```
git good / git bad
```

c. 复杂情况

有时还有其他复杂情况，例如当不同修订版的依赖关系不相同时，例如一个修订版可能需要 `numpy=1.25`，而另一个需要 `numpy=1.26`。如果依赖包版本向后兼容，安装较新的版本应该可以解决问题。但情况并非总是如此。因此，有时在重新测试程序之前，必须重新安装正确的依赖项。

有时，当有一系列实际上以不同方式损坏的提交时，它会有所帮助，您可以找到一个不包含其他坏范围的 `good...bad` 提交范围，或者您可以尝试如前所述 `git bisect skip` 其他坏提交。
