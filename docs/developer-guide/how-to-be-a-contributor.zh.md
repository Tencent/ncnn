### 如何提交代码

#### 一、fork 分支
在浏览器中打开 [ncnn](github.com/tencent/ncnn), `fork` 到自己的 repositories，例如
```
github.com/user/ncnn
```
clone 项目，fetch 官方 remote

```
$ git clone https://github.com/user/ncnn && cd ncnn
$ git remote add tencent https://github.com/tencent/ncnn
$ git fetch tencent
$ git checkout tencent/master
```
创建自己的分支，命名尽量言简意赅。一个分支只做一件事，方便 review 和 revert。例如：
```
$ git checkout -b add-conv-int8
```

#### 二、代码习惯
为了增加沟通效率，reviewer 一般要求 contributor 遵从以下规则

* `if-else`和花括号`{`中间需要换行
* 不能随意增删空行
* tab 替换为 4 个空格
* 为了保证平台兼容性，目前不使用`c++11`，`src`目录下尽量避免使用`template`
* 若是新增功能或平台，`test`目录需有对应测试用例

开发完成后提交到自己的 repository
```
$ git push origin add-conv-int8
```

#### 三、代码提交
浏览器中打开 github.com/user/ncnn