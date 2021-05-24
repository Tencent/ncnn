### 如何提交代码

#### 一、fork 分支
在浏览器中打开 [ncnn](https://github.com/tencent/ncnn), `fork` 到自己的 repositories，例如
```
https://github.com/user/ncnn
```

clone 项目到本地，添加官方 remote 并 fetch:
```
$ git clone https://github.com/user/ncnn && cd ncnn
$ git remote add tencent https://github.com/tencent/ncnn
$ git fetch tencent
```
对于 `git clone` 下来的项目，它现在有两个 remote，分别是 origin 和 tencent：

```
$ git remote -v
origin   https://github.com/user/ncnn (fetch)
origin   https://github.com/user/ncnn (push)
tencent  https://github.com/Tencent/ncnn (fetch)
tencent  https://github.com/Tencent/ncnn (push)
```
origin 指向你 fork 的仓库地址；remote 即官方 repo。可以基于不同的 remote 创建和提交分支。

例如切换到官方 master 分支，并基于此创建自己的分支（命名尽量言简意赅。一个分支只做一件事，方便 review 和 revert）
```
$ git checkout tencent/master
$ git checkout -b add-conv-int8
```

或创建分支时指定基于官方 master 分支：
```
$ git checkout -b fix-typo-in-document tencent/master
```

> `git fetch` 是从远程获取最新代码到本地。如果是第二次 pr ncnn，直接从  `git fetch tencent` 开始即可，不需要 `git remote add tencent`，也不需要修改 `github.com/user/ncnn`。

#### 二、代码习惯
为了增加沟通效率，reviewer 一般要求 contributor 遵从以下规则

* `if-else`和花括号`{`中间需要换行
* 不能随意增删空行
* tab 替换为 4 个空格
* 为了保证平台兼容性，目前不使用`c++11`，`src`目录下尽量避免使用`template`
* 若是新增功能或平台，`test`目录需有对应测试用例
* 文档放到`doc`对应目录下，中文用`.zh.md`做后缀；英文直接用`.md`后缀

开发完成后提交到自己的 repository
```
$ git commit -a
$ git push origin add-conv-int8
```
推荐使用 [`commitizen`](https://pypi.org/project/commitizen/) 或 [`gitlint`](https://jorisroovers.com/gitlint/) 等工具格式化 commit message，方便事后检索海量提交记录

#### 三、代码提交
浏览器中打开 [ncnn pulls](https://github.com/Tencent/ncnn/pulls) ，此时应有此分支 pr 提示，点击 `Compare & pull request`

* 标题**必须**是英文。未完成的分支应以 `WIP:` 开头，例如 `WIP: add conv int8`
* 正文宜包含以下内容，中英不限
    * 内容概述和实现方式
    * 功能或性能测试
    * 测试结果

CI 已集成了自动格式化，restyled-io 会在 pr 的同时生成 `Restyled add conv int8`，需要 merge 自动 restyled 的分支，例如
```
$ git fetch tencent
$ git checkout add-conv-int8
$ git merge tencent/restyled/pull-2078
$ git push origin add-conv-int8
```
回到浏览器签署  CLA，所有 CI 测试通过后通知 reviewer merge 此分支。

#### 四、彩蛋
留下个人 qq 号会触发隐藏事件。