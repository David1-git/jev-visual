from jev_visual.adapters_torch_finetuned import (
    load_finetuned_adapter,
    process_request_file,
    process_structured_request
)

# 方法1: 直接从文件
results = process_request_file('examples/trash-overflow-request.json', model_path=model_path)

# 方法2: 手动控制
adapter, _, _ = load_finetuned_adapter(model_path)
results = process_structured_request(request, adapter)