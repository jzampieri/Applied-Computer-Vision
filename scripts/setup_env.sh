#!/usr/bin/env bash
# Cria o ambiente virtual e instala as dependencias (CPU-only).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
echo "==> Python: $($PY --version)"

if [ ! -d .venv ]; then
  echo "==> Criando .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel

echo "==> Instalando requirements (indice CPU do PyTorch)"
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# pytorchvideo: o release do PyPI (0.1.5, 2022) importa torchvision.transforms.functional_tensor,
# removido no torchvision moderno. A main do repo ja esta corrigida. Instalamos sem deps para
# nao deixar o resolvedor rebaixar torch/torchvision.
echo "==> Instalando pytorchvideo (git main, --no-deps)"
pip install --no-deps "git+https://github.com/facebookresearch/pytorchvideo.git"

echo "==> Smoke test"
python scripts/smoke_test.py
