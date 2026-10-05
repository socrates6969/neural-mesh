import os, sys, json, time, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
SP=os.path.dirname(os.path.abspath(__file__)); M=os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2")
tok=Tokenizer.from_file(os.path.join(M,"tokenizer.json")); tok.no_padding(); tok.enable_truncation(max_length=128)
V={"fp32":os.path.join(M,"model.onnx"),"q8":f"{SP}/model_q8.onnx","t5":f"{SP}/trunc5.onnx","t4":f"{SP}/trunc4.onnx","t3":f"{SP}/trunc3.onnx","t4q8":f"{SP}/trunc4_q8.onnx"}
def sess(path,thr):
    so=ort.SessionOptions(); so.intra_op_num_threads=thr; so.inter_op_num_threads=1; return ort.InferenceSession(path,so,providers=["CPUExecutionProvider"])
def enc1(s,names,text):
    e=tok.encode(text); ids=np.array([e.ids],np.int64); am=np.ones_like(ids)
    feed={"input_ids":ids,"attention_mask":am}
    if "token_type_ids" in names: feed["token_type_ids"]=np.zeros_like(ids)
    h=s.run(None,feed)[0][0]; p=h.mean(0); return p/max(np.linalg.norm(p),1e-12)
q=json.load(open(f"{SP}/qtexts.json")); allq=q["A"]["text"]+q["B"]["text"]
thr=int(sys.argv[1]) if len(sys.argv)>1 else 1
res={}
for name,path in V.items():
    s=sess(path,thr); names=[i.name for i in s.get_inputs()]
    for t in allq[:20]: enc1(s,names,t)  # warm
    E=np.zeros((len(allq),384),np.float32); lat=[]
    for i,t in enumerate(allq):
        a=time.perf_counter(); E[i]=enc1(s,names,t); lat.append((time.perf_counter()-a)*1e3)
    np.save(f"{SP}/E_{name}.npy",E); res[name]={"p50_ms":float(np.percentile(lat,50)),"p90_ms":float(np.percentile(lat,90)),"mean_ms":float(np.mean(lat))}
    print(name,thr,res[name],flush=True)
json.dump(res,open(f"{SP}/lat_t{thr}.json","w"))
