## 只是实践经验，没有理论，不一定正确

```
prfm pldl1keep, [x0, #256]
```
* 放在 ld1 [x0] 前面 0~8 条指令
* #256 表示把 x0+256 的内容放进 L1 cache
* ldp 也适用
* (经验)不写 offset 不如写个 #128
* (经验)pldl1strm 似乎没啥意思，也没 pldl1keep 快
* (经验)x0 ~ x0+256 的内容也会进来
* (经验)load 128bit 用 #128，256bit或更多用 #256
* (经验)避免 pld a，pld b，load a，load b 顺序，可能相互干扰
* (经验)提前太多会失效
* (经验)适合连续读

```
prfm pldl2strm, [x0, #256]
```
* 放在 ld1 [x0] 前面 N 条指令，N 尽量大些
* #256 表示把 x0+256 的内容放进 L2 cache
* ldp 也适用
* (经验)不写 offset 不如写个 #128
* (经验)pldl2strm 效果稍好于 pldl2keep
* (经验)x0 ~ x0+256 的内容也会进来
* (经验)load 128bit 用 #128，256bit 用 #256
* (经验)读很多数据，用不同 offset 连续两次 pldl2strm
* (经验)后面不要对同位置再 pldl1keep，会变慢
* (经验)适合提前准备要跳到很远的地方读，比如换 channel
