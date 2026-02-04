# API 参考

本节将介绍 THETA 的主要 API 接口及其用法。

## 推理接口
- 输入模型参数，返回推理结果。
- 支持多种模型格式。

## 示例代码
```python
# 示例：调用推理接口
from theta import InferenceEngine
engine = InferenceEngine(model_path='your_model_path')
result = engine.infer(input_data)
print(result)
```

## 更多接口
请参考项目源码及 README 文档。

---

如需更多示例，请查阅 [快速开始](quickstart.md)。