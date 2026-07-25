# Export subsystem rules

- Start with a static input shape and preserve raw semantic logits.
- Remove or name auxiliary outputs explicitly; never guess which tensor is primary.
- Every export requires numerical and task-level equivalence evidence.
- Do not invent ONNX Runtime or TensorRT results.
- TensorRT engines are device/version specific and are built on the target Jetson.
