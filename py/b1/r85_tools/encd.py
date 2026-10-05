import os, sys, json, time, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
SP=os.path.dirname(os.path.abspath(__file__)); M=os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2")
tok=Tokenizer.from_file(os.path.join(M,"tokenizer.json")); tok.no_padding(); tok.enable_truncation(max_length=128)
V={"q8":f"{SP}/model_q8.onnx","t4":f"{SP}/trunc4.onnx","t3":f"{SP}/trunc3.onnx","t4q8":f"{SP}/trunc4_q8.onnx"}
D=json.load(open(f"{SP}/dtexts.json"))
for name,path in V.items():
    so=ort.SessionOptions(); so.intra_op_num_threads=8; s=ort.InferenceSession(path,so,providers=["CPUExecutionProvider"]); names=[i.name for i in s.get_inputs()]
    for tag in "AB":
        texts=D[tag]; enc=tok.encode_batch(texts); order=sorted(range(len(texts)),key=lambda i:len(enc[i].ids)); out=np.zeros((len(texts),384),np.float32); t0=time.time()
        for st in range(0,len(order),64):
            idx=order[st:st+64]; L=max(len(enc[i].ids) for i in idx); ids=np.zeros((len(idx),L),np.int64); am=np.zeros((len(idx),L),np.int64)
            for r,i in enumerate(idx): n=len(enc[i].ids); ids[r,:n]=enc[i].ids; am[r,:n]=1
            feed={"input_ids":ids,"attention_mask":am}
            if "token_type_ids" in names: feed["token_type_ids"]=np.zeros_like(ids)
            h=s.run(None,feed)[0]; m=am[:,:,None].astype(np.float32); p=(h*m).sum(1)/np.clip(m.sum(1),1e-9,None); p/=np.clip(np.linalg.norm(p,axis=1,keepdims=True),1e-12,None)
            for r,i in enumerate(idx): out[i]=p[r]
        np.save(f"{SP}/D_{name}_{tag}.npy",out); print(name,tag,len(texts),round(time.time()-t0,1),"s",flush=True)
