### broadcasting rule

ncnn BinaryOp accepts blobs with different shape

C = BinaryOp(A, B)

shape notation convention is [w], [w,h], [w,h,c]

Two tensors are “broadcastable” if the following rules hold:

- Each tensor has at least one dimension.
- When iterating over the dimension sizes, starting at the last dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

|type|A|B|C|
|---|---|---|---|
|1|[2,3,4]|[2,3,1]|[2,3,4]|
|1|[2,1,1]|[1,3,4]|[2,3,4]|
|1|[1,3,1]|[2,1,4]|[2,3,4]|
|1|[2,1,4]|[1,1,1]|[2,1,4]|
|1|...|...|...|
|2|[2,3,4]|[3,4]|[2,3,4]|
|2|[2,3,4]|[3,1]|[2,3,4]|
|2|[2,1,4]|[3,1]|[2,3,4]|
|2|...|...|...|
|3|[2,3,4]|[4]|[2,3,4]|
|3|[2,3,1]|[4]|[2,3,4]|
|3|...|...|...|
|4|[3,1]|[2,3,4]|[2,3,4]|
|4|[1,4]|[2,3,4]|[2,3,4]|
|4|...|...|...|
|5|[3,4]|[3,1]|[3,4]|
|5|[3,1]|[1,4]|[3,4]|
|5|...|...|...|
|6|[3,4]|[4]|[3,4]|
|6|...|...|...|
|7|[4]|[2,3,4]|[2,3,4]|
|7|...|...|...|
|8|[4]|[3,4]|[3,4]|
|8|...|...|...|
|9|[4]|[4]|[4]|
|9|...|...|...|

cause ncnn use the layout [w,h,c], it will align from the last dimension when broadcasting (namely `c` when `dims=3`, `h` when `dims=2`, `w` when `dims=1`). But some framework like pytorch prefer using the [b,c,h,w] layout, so when convert to ncnn, it will align from the first dimension when boradcasting (always `w`). So we add a layout flag for BinaryOp to control the align direction.

- `0` default option, [w,h,c] layout, will align from the last dimension to the first dimension, c --> h --> w 
- `1` [c,h,w] layout, when convert to ncnn, will align from the first dimension to the last dimension, w --> h --> c

ps: the table above only demonstrate the [w,h,c] layout broadcasting 
