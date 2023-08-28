# python pnnx wrapper

## how to use?

```python
import pnnx
x = torch.rand(1,3,224,224)
y = torch.rand(1,3)
pnnx.export(model, (x, y))
```