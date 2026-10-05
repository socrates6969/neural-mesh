import onnx, os, sys
from onnx.utils import extract_model
from onnxruntime.quantization import quantize_dynamic, QuantType
src=os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2/model.onnx"); out=os.path.dirname(os.path.abspath(__file__))
m=onnx.load(src)
# find per-layer output tensor names
def lay_out(k):
    for n in m.graph.node:
        if n.name==f"/encoder/layer.{k}/output/LayerNorm/Add_1": return n.output[0]
for k in (5,4,3,2):
    if k==5: continue
    dst=os.path.join(out,f"trunc{k+1}.onnx"); extract_model(src,dst,["input_ids","attention_mask","token_type_ids"],[lay_out(k)]); print("wrote",dst)
for name in ("model","trunc4"):
    s=src if name=="model" else os.path.join(out,"trunc4.onnx"); d=os.path.join(out,f"{name}_q8.onnx")
    quantize_dynamic(s,d,weight_type=QuantType.QInt8); print("wrote",d, os.path.getsize(d)/1e6)
