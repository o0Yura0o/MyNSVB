import pickle
import numpy as np

path = "runs/raw_infer_20251113_042914/bin/test.data"

with open(path, "rb") as f:
    try:
        while True:
            obj = pickle.load(f)
            print("--- Item ---")
            for k, v in obj.items():
                print(f"{k}: {type(v)} {getattr(v, 'shape', '')}")
                if k == "multi_spk_emb":
                    print(v[:,0,:])
    except EOFError:
        print("End of file reached.")

# with open(path, "rb") as f:
#     idx = 0
#     try:
#         while True:
#             item = pickle.load(f)
#             if "multi_spk_emb" in item:
#                 emb = item["multi_spk_emb"]
#                 print(f"--- Item #{idx} ---")
#                 print(f"multi_spk_emb shape: {emb.shape}")
#                 # 印前兩個 embedding 的前 10 維
#                 for i, e in enumerate(emb[:2]):
#                     print(f"  Emb[{i}] first 10 dims: {np.round(e[:10], 4)}")
#                 print()
#             idx += 1
#     except EOFError:
#         print("End of file reached.")