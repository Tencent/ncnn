### broadcasting rule

ncnn BinaryOp accepts blobs with different shape

C = BinaryOp(A, B)

shape notation convention is [w], [w,h], [w,h,c], [w,h,d,c]

* binaryop with scalar and scalar-like

|A|B|C|
|---|---|---|
|[2]|scalar / [1]|[2]|
|[2,3]|scalar / [1] / [1,1]|[2,3]|
|[2,3,4]|scalar / [1] / [1,1] / [1,1,1]|[2,3,4]|
|[2,3,4,5]|scalar / [1] / [1,1] / [1,1,1] / [1,1,1,1]|[2,3,4,5]|

* no broadcast

|A|B|C|
|---|---|---|
|[2]|[2]|[2]|
|[2,3]|[2,3]|[2,3]|
|[2,3,4]|[2,3,4]|[2,3,4]|
|[2,3,4,5]|[2,3,4,5]|[2,3,4,5]|

* explicit broadcast B

|A|B|C|
|---|---|---|
|[2,3]|[1,3]|[2,3]|
|[2,3]|[2,1]|[2,3]|
|[2,3,4]|[1,3,4]|[2,3,4]|
|[2,3,4]|[2,1,4]|[2,3,4]|
|[2,3,4]|[2,3,1]|[2,3,4]|
|[2,3,4]|[1,1,4]|[2,3,4]|
|[2,3,4]|[1,3,1]|[2,3,4]|
|[2,3,4]|[2,1,1]|[2,3,4]|
|[2,3,4,5]|[1,3,4,5]|[2,3,4,5]|
|[2,3,4,5]|[2,1,4,5]|[2,3,4,5]|
|[2,3,4,5]|[2,3,1,5]|[2,3,4,5]|
|[2,3,4,5]|[2,3,4,1]|[2,3,4,5]|
|[2,3,4,5]|[1,1,4,5]|[2,3,4,5]|
|[2,3,4,5]|[1,3,1,5]|[2,3,4,5]|
|[2,3,4,5]|[1,3,4,1]|[2,3,4,5]|
|[2,3,4,5]|[2,1,1,5]|[2,3,4,5]|
|[2,3,4,5]|[2,1,4,1]|[2,3,4,5]|
|[2,3,4,5]|[2,3,1,1]|[2,3,4,5]|
|[2,3,4,5]|[1,1,1,5]|[2,3,4,5]|
|[2,3,4,5]|[1,1,4,1]|[2,3,4,5]|
|[2,3,4,5]|[1,3,1,1]|[2,3,4,5]|
|[2,3,4,5]|[2,1,1,1]|[2,3,4,5]|

* implicit broadcast B for inner axis

It broadcasts in the opposite direction of the numpy's implicit broadcasting behavior.

pnnx will insert reshape operator at the appropriate position to convert it to explicit broadcast automatically.

|A|B|C|
|---|---|---|
|[2,3]|[3]|[2,3]|
|[2,3,4]|[4]|[2,3,4]|
|[2,3,4]|[3,4]|[2,3,4]|
|[2,3,4,5]|[5]|[2,3,4,5]|
|[2,3,4,5]|[4,5]|[2,3,4,5]|
|[2,3,4,5]|[3,4,5]|[2,3,4,5]|

* implicit broadcast B with 1 dimension rank for outer axis

This exists only for compatibility.

When the size is the same, eg. [2,2] and [2], broadcast B for inner axis will be prioritized.

|A|B|C|
|---|---|---|
|[2,3]|[2]|[2,3]|
|[2,3,4]|[2]|[2,3,4]|
|[2,3,4,5]|[2]|[2,3,4,5]|
