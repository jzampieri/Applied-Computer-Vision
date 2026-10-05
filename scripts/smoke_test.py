"""Verifica que a stack inteira importa e roda um forward em CPU.

Gate da fase 1 do plano: se o SlowFast do pytorchvideo nao carregar neste
ambiente (Python 3.14 + torch 2.14), caimos no modelo vendorizado em
src/slowfast_model.py.
"""

import sys


def main() -> int:
    import numpy as np
    import torch

    print(f"torch       {torch.__version__}  (cuda={torch.cuda.is_available()})")
    print(f"numpy       {np.__version__}")

    import cv2

    print(f"opencv      {cv2.__version__}")

    import onnxruntime as ort

    print(f"onnxruntime {ort.__version__}  providers={ort.get_available_providers()}")

    import ultralytics

    print(f"ultralytics {ultralytics.__version__}")

    ok = True
    try:
        from pytorchvideo.models.hub import slowfast_r50

        model = slowfast_r50(pretrained=False).eval()
        # SlowFast recebe uma lista [slow, fast]: slow = fast subamostrado por alpha=4.
        fast = torch.randn(1, 3, 32, 256, 256)
        slow = fast[:, :, ::4]
        with torch.no_grad():
            out = model([slow, fast])
        print(f"pytorchvideo OK -> logits {tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001 - o gate precisa reportar qualquer falha
        ok = False
        print(f"pytorchvideo FALHOU: {type(exc).__name__}: {exc}")
        print("  -> usar o caminho B (src/slowfast_model.py vendorizado)")

    print("\nSMOKE TEST:", "OK" if ok else "PARCIAL (ver acima)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
