# Procedencia e licenca dos videos

Todos os videos usados no CP5 sao de dominio publico ou licenca Creative Commons.
Nenhum video foi editado alem do recorte temporal e do redimensionamento
(640 px de largura, 30 fps) feitos por
`scripts/fetch_videos.py`.

## `archery.mp4`

- **Papel no experimento:** Caso limpo: uma pessoa, acao presente no vocabulario Kinetics-400.
- **Acao observada (verdade de referencia):** archery
- **Autor / creditos:** PyTorchVideo / Facebook AI Research
- **Licenca:** Amostra de pesquisa do PyTorchVideo (clipe do Kinetics-400)
- **Pagina da fonte:** https://pytorchvideo.org/
- **URL do arquivo:** https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4
- **Recorte aplicado:** inicio 0.0s, duracao 10.0s

## `pushups.mp4`

- **Papel no experimento:** Fronteiras de acao: series alternadas com descanso, para medir se a confianca cai ou a classe muda quando a acao comeca e termina.
- **Acao observada (verdade de referencia):** push up
- **Autor / creditos:** Wikimedia Commons
- **Licenca:** CC BY-SA 4.0
- **Pagina da fonte:** https://commons.wikimedia.org/wiki/File:Interval_Push-ups.webm
- **URL do arquivo:** https://upload.wikimedia.org/wikipedia/commons/9/98/Interval_Push-ups.webm
- **Recorte aplicado:** inicio 2.0s, duracao 12.0s

## `skate.mp4`

- **Papel no experimento:** Casos de falha: multiplas pessoas, oclusao mutua, movimento rapido e sujeitos pequenos no quadro.
- **Acao observada (verdade de referencia):** skateboarding
- **Autor / creditos:** Wikimedia Commons
- **Licenca:** CC BY-SA 4.0
- **Pagina da fonte:** https://commons.wikimedia.org/wiki/File:Day_of_the_Town_(2017-09-09)_-_Skateboarding_-_0022.webm
- **URL do arquivo:** https://upload.wikimedia.org/wikipedia/commons/4/4d/Day_of_the_Town_%282017-09-09%29_-_Skateboarding_-_0022.webm
- **Recorte aplicado:** inicio 0.0s, duracao 12.0s
