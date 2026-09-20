"""探测一个 conda 环境里是否具备 UniChess 需要的依赖。只读，不安装任何东西。"""
import sys

print("  python", sys.version.split()[0])
for m in ("torch", "numpy", "chess", "zstandard"):
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "?")
        print("  {:10} {}".format(m, v))
    except ImportError:
        print("  {:10} 缺失".format(m))
try:
    import torch
    ok = torch.cuda.is_available()
    print("  cuda:", ok)
    if ok:
        print("  device:", torch.cuda.get_device_name(0))
        print("  capability: sm_%d%d" % torch.cuda.get_device_capability(0))
        print("  arch_list:", torch.cuda.get_arch_list())
except Exception as e:
    print("  torch 异常:", type(e).__name__, e)
