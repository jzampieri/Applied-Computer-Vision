# CP5 - Deteccao de Acoes de Pessoas com YOLO + U-Net + SlowFast
PY := .venv/bin/python

.PHONY: help setup data run analyze report all clean-outputs

help:
	@echo "make setup    - cria .venv, instala dependencias e roda o smoke test"
	@echo "make data     - baixa pesos pre-treinados e videos (CC/dominio publico)"
	@echo "make run      - roda o pipeline completo nos tres videos"
	@echo "make analyze  - metricas, momentos, paineis e graficos"
	@echo "make report   - gera report/CP5_relatorio.pdf"
	@echo "make all      - data + run + analyze + report"

setup:
	bash scripts/setup_env.sh

data:
	$(PY) scripts/fetch_weights.py
	$(PY) scripts/fetch_videos.py

run:
	$(PY) -m src.cli all --video all

analyze:
	$(PY) -m src.cli analyze

report:
	$(PY) -m src.cli report

all: data run analyze report

clean-outputs:
	rm -rf outputs report/panels report/metrics.csv report/summary.json
