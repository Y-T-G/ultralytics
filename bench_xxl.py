import sys, time, torch
from ultralytics.nn.tasks import DetectionModel
dev = f"cuda:{sys.argv[1]}"; ITERS, WARM = 60, 20
def timeit(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(dev); t = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize(dev); return (time.perf_counter()-t)/ITERS*1000
def feats(m, x):
    o, y = [], x
    for l in m.model[:-1]:
        y = (y if l.f == -1 else o[l.f]) if isinstance(l.f, int) else [y if j == -1 else o[j] for j in l.f]
        y = l(y); o.append(y)
    return [o[j] for j in m.model[-1].f]
x = torch.rand(1, 3, 640, 640, device=dev, dtype=torch.half)
print(f"{'model':10s} {'head M':>7s} {'full ms':>8s} {'head ms':>8s}")
ck = torch.load("/ultralytics/yolo27xxl-detr-flat.pt", map_location="cpu", weights_only=False)
t = (ck.get("ema") or ck["model"]).float().fuse().eval().half().to(dev)
with torch.inference_mode():
    hp = sum(p.numel() for p in t.model[-1].parameters())/1e6
    fu = timeit(lambda: t(x)); inp = feats(t, x)
    print(f"{'DeimDecoder':10s} {hp:7.2f} {fu:8.2f} {timeit(lambda: t.model[-1](inp)):8.2f}", flush=True)
del t; torch.cuda.empty_cache()
for n in sys.argv[2:]:
    m = DetectionModel(f"/ultralytics/configs/head/yolo27xxl-det-{n}.yaml", nc=80, verbose=False).fuse(verbose=False).eval().half().to(dev)
    with torch.inference_mode():
        hp = sum(p.numel() for p in m.model[-1].parameters())/1e6
        fu = timeit(lambda: m(x)); inp = feats(m, x)
        print(f"{n:10s} {hp:7.2f} {fu:8.2f} {timeit(lambda: m.model[-1](inp)):8.2f}", flush=True)
    del m; torch.cuda.empty_cache()
