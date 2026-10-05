"""Texto analitico do relatorio.

Os numeros citados aqui foram medidos pelo pipeline e estao em
report/summary.json e report/metrics.csv. As tabelas do PDF sao geradas
dinamicamente a partir desses arquivos; este modulo guarda apenas a analise.
"""

TITULO = "CP5 \u2014 Detec\u00e7\u00e3o de A\u00e7\u00f5es de Pessoas com YOLO, U-Net e SlowFast"

SUBTITULO = (
    "Applied Computer Vision 2026 \u2014 CheckPoint 5 \u00b7 Prof. Dr. Paulo Sergio Rodrigues \u00b7 "
    "Modelos pr\u00e9-treinados integrados, sem treinamento adicional."
)

# TODO(grupo): substituir nome, RM e parecer de cada integrante.
MEMBROS = [
    {"nome": "[Nome do integrante 1]", "rm": "[RM]",
     "responsabilidade": "YOLO e leitura do v\u00eddeo",
     "parecer": "[parecer individual sobre os resultados]"},
    {"nome": "[Nome do integrante 2]", "rm": "[RM]",
     "responsabilidade": "U-Net e proje\u00e7\u00e3o da m\u00e1scara",
     "parecer": "[parecer individual sobre os resultados]"},
    {"nome": "[Nome do integrante 3]", "rm": "[RM]",
     "responsabilidade": "SlowFast e janelas temporais",
     "parecer": "[parecer individual sobre os resultados]"},
    {"nome": "[Nome do integrante 4]", "rm": "[RM]",
     "responsabilidade": "Integra\u00e7\u00e3o, testes e apresenta\u00e7\u00e3o",
     "parecer": "[parecer individual sobre os resultados]"},
]

OBJETIVO = (
    "O programa l\u00ea um v\u00eddeo curto e produz, para cada trecho, tr\u00eas sa\u00eddas sobrepostas no "
    "mesmo quadro: a caixa da pessoa, os pixels que pertencem a ela e a a\u00e7\u00e3o que ela realiza. "
    "Os tr\u00eas modelos s\u00e3o pr\u00e9-treinados e nenhum recebeu treinamento adicional. O pipeline roda "
    "em quatro est\u00e1gios independentes que gravam artefatos em disco \u2014 o enunciado pede que os "
    "modelos sejam executados separadamente, e a separa\u00e7\u00e3o tamb\u00e9m mant\u00e9m um \u00fanico modelo por "
    "vez na mem\u00f3ria, o que foi necess\u00e1rio porque a m\u00e1quina de refer\u00eancia n\u00e3o tem GPU e disp\u00f5e "
    "de 6 GB de RAM."
)

MODELOS = [
    ["Detec\u00e7\u00e3o",
     "YOLO11n (<i>ultralytics</i>), classe <i>person</i>, tracker ByteTrack",
     "Localiza cada pessoa e mant\u00e9m um <i>track_id</i> est\u00e1vel ao longo do tempo."],
    ["Segmenta\u00e7\u00e3o",
     "U\u00b2-Net <i>u2net_human_seg</i> (ONNX Runtime)",
     "Produz a m\u00e1scara da pessoa dentro do recorte da caixa. U\u00b2-Net \u00e9 da fam\u00edlia U-Net "
     "(<i>encoder-decoder</i> com <i>skip connections</i>, blocos RSU em estrutura U aninhada) e "
     "este checkpoint \u00e9 pr\u00e9-treinado para segmenta\u00e7\u00e3o de pessoas."],
    ["A\u00e7\u00e3o",
     "SlowFast R50 8\u00d78 (<i>pytorchvideo</i>), Kinetics-400",
     "Classifica a a\u00e7\u00e3o em janelas de 32 quadros com passo 16, registrando top-1, top-5 e "
     "confian\u00e7a por janela."],
]

INTEGRACAO = (
    "<b>Integra\u00e7\u00e3o YOLO + U-Net.</b> O recorte da caixa \u00e9 expandido em 10% antes de ir para a "
    "U-Net; sem essa folga o contorno sai cortado na borda do recorte, porque a rede n\u00e3o enxerga "
    "o entorno do sujeito. A m\u00e1scara volta ao tamanho do recorte e \u00e9 colada na posi\u00e7\u00e3o original "
    "do quadro. <b>Integra\u00e7\u00e3o com o SlowFast.</b> Cada janela recebe os 32 quadros no <i>fast "
    "pathway</i> e a subamostragem por <i>alpha</i>=4 no <i>slow pathway</i>. Com <i>roi=pessoa</i> "
    "a janela \u00e9 recortada na uni\u00e3o das caixas, de modo que o sujeito preencha o quadro; com "
    "<i>roi=quadro</i> usa-se a imagem inteira. Cada quadro herda a a\u00e7\u00e3o da janela que o cont\u00e9m "
    "e, nas regi\u00f5es de sobreposi\u00e7\u00e3o, vence a janela de maior confian\u00e7a."
)

LEGENDA_TABELA = (
    "Cobertura YOLO = fra\u00e7\u00e3o de quadros com ao menos uma pessoa detectada. IoU temp. = IoU m\u00e9dia "
    "entre as caixas do sujeito principal em quadros consecutivos (mede o quanto a caixa acompanha "
    "a pessoa). Trocas do sujeito principal = quantas vezes a maior caixa muda de track_id; com "
    "duas pessoas de tamanho parecido isso alterna sem que o rastreador tenha errado. Fill ratio = "
    "\u00e1rea da m\u00e1scara dividida pela \u00e1rea da caixa, com m\u00ednimo e m\u00e1ximo entre par\u00eanteses. Acertos "
    "e confian\u00e7a do SlowFast referem-se a <i>roi=pessoa</i>."
)

CUSTO = (
    "Medido na m\u00e1quina de refer\u00eancia: Ubuntu 26.04, Python 3.14, 8 n\u00facleos, 6 GB de RAM, "
    "<b>sem GPU</b>. V\u00eddeos normalizados para 640 px e 30 fps. A coluna do SlowFast \u00e9 o tempo de "
    "uma passagem; o total considera as duas ROIs (pessoa e quadro inteiro). A segmenta\u00e7\u00e3o domina "
    "o custo porque roda uma vez por pessoa por quadro \u2014 \u00e9 o est\u00e1gio a otimizar primeiro."
)

PERGUNTAS_INTRO = (
    "Os momentos abaixo n\u00e3o foram escolhidos a esmo: al\u00e9m de in\u00edcio, meio e fim de cada v\u00eddeo, o "
    "analisador inclui automaticamente todas as janelas em que o top-1 do SlowFast muda e o quadro "
    "de menor \u00e1rea de m\u00e1scara \u2014 \u00e9 exatamente onde o pipeline falha, e \u00e9 isso que interessa "
    "examinar."
)

RESPOSTAS = [
    ("A caixa acompanha a pessoa?",
     "Sim, nos dois v\u00eddeos exigidos. A cobertura \u00e9 de 100% dos quadros em <i>archery</i> e "
     "<i>pushups</i>, com IoU temporal m\u00e9dia de 0,97 e 0,99 \u2014 a caixa gruda no sujeito mesmo "
     "durante o movimento r\u00e1pido da flex\u00e3o. O v\u00eddeo extra <i>skate</i> \u00e9 o contraexemplo: a "
     "cobertura segue alta (99,4%), mas a IoU temporal cai para 0,77 e o rastreador emite 124 "
     "identidades para um punhado de pessoas reais (erro 3). Localizar \u00e9 f\u00e1cil; manter a "
     "identidade, n\u00e3o."),
    ("A m\u00e1scara acompanha o contorno?",
     "Parcialmente, e \u00e9 o elo mais fr\u00e1gil do pipeline. Em <i>pushups</i> a m\u00e1scara \u00e9 est\u00e1vel "
     "(varia\u00e7\u00e3o m\u00e9dia de \u00e1rea entre quadros de apenas 1,6%) mas sistematicamente incompleta: o "
     "<i>fill ratio</i> fica entre 0,25 e 0,50 porque bra\u00e7os e cabe\u00e7a, finos e em contraluz, ficam "
     "de fora. Em <i>archery</i> a m\u00e1scara \u00e9 boa na maior parte do v\u00eddeo e colapsa por completo no "
     "trecho final (erro 1). Em <i>skate</i> o quadro \u00e9 brutal: 68,9% das caixas saem sem m\u00e1scara "
     "nenhuma e o <i>fill ratio</i> m\u00e9dio \u00e9 de apenas 0,06."),
    ("A a\u00e7\u00e3o prevista corresponde ao trecho?",
     "Em <i>archery</i>, sim: 100% das 18 janelas classificam <i>archery</i> com confian\u00e7a 1,00 "
     "\u2014 o v\u00eddeo \u00e9 um clipe do pr\u00f3prio Kinetics-400, ent\u00e3o funciona como verifica\u00e7\u00e3o de "
     "sanidade do pr\u00e9-processamento. Em <i>pushups</i> o top-1 acerta 82% das janelas e o top-5, "
     "95%; os erros se concentram nos dois primeiros segundos, em que a pessoa ainda est\u00e1 se "
     "posicionando e o modelo responde <i>yoga</i>. Em <i>skate</i> o top-1 acerta 68% e o top-5 "
     "acerta <b>100%</b>: a classe certa est\u00e1 sempre entre as cinco primeiras, mas disputa com "
     "<i>high jump</i>, <i>gymnastics tumbling</i> e <i>waiting in line</i> \u2014 classes que "
     "descrevem bem o que mais ocupa a cena, que \u00e9 gente parada assistindo."),
    ("A confian\u00e7a cai ou a classe muda quando a a\u00e7\u00e3o come\u00e7a ou termina?",
     "Sim, e de forma mensur\u00e1vel. Em <i>pushups</i> a confian\u00e7a m\u00e9dia nas janelas de troca de "
     "classe \u00e9 0,75, contra 0,82 nas est\u00e1veis; em <i>skate</i> a margem \u00e9 ainda maior, 0,49 "
     "contra 0,79. O gr\u00e1fico de <i>pushups</i> mostra o padr\u00e3o completo: a curva parte de "
     "<i>yoga</i> na fase de prepara\u00e7\u00e3o, sobe para 1,00 entre 3,7 s e 5,9 s \u2014 quando as "
     "repeti\u00e7\u00f5es se sucedem sem pausa \u2014 e cai para o m\u00ednimo do v\u00eddeo, 0,44, por volta de 9 s. "
     "Inspecionando esse trecho, o m\u00ednimo n\u00e3o \u00e9 um erro de pose: o atleta <b>sustenta a posi\u00e7\u00e3o "
     "alta</b>, de bra\u00e7os estendidos, por mais de um segundo. A postura continua sendo a de uma "
     "flex\u00e3o, mas o movimento praticamente cessa, e \u00e9 o movimento que o <i>fast pathway</i> "
     "codifica. Em <i>archery</i>, onde a a\u00e7\u00e3o \u00e9 cont\u00ednua do primeiro ao \u00faltimo quadro, n\u00e3o h\u00e1 "
     "nenhuma troca de classe nem queda de confian\u00e7a: o contraste entre os v\u00eddeos \u00e9 a resposta."),
]

ERROS = [
    {
        "titulo": "Erro 1 \u2014 A m\u00e1scara da U-Net colapsa enquanto o YOLO continua correto "
                  "(<i>archery</i>, t = 9,7 s).",
        "texto": (
            "No quadro 291 o YOLO detecta as duas arqueiras com confian\u00e7a 0,83 e 0,84, o SlowFast "
            "mant\u00e9m <i>archery</i> a 1,00, e a m\u00e1scara simplesmente desaparece: restam fragmentos "
            "que cobrem 0,2% do recorte, contra 27,8% em um quadro bom do mesmo v\u00eddeo. "
            "<b>Diagn\u00f3stico medido:</b> a sali\u00eancia bruta da U\u00b2-Net cai de m\u00e1ximo 1,000 e m\u00e9dia "
            "0,262 para m\u00e1ximo 0,646 e m\u00e9dia 0,008. Baixar o limiar de 0,5 para 0,1 recupera apenas "
            "3,6% do recorte (contra 32,2% no quadro bom), ent\u00e3o n\u00e3o \u00e9 problema de limiariza\u00e7\u00e3o "
            "\u2014 \u00e9 a pr\u00f3pria resposta da rede que se anula. As duas caixas n\u00e3o se sobrep\u00f5em "
            "(IoU 0,000) e as propor\u00e7\u00f5es do recorte s\u00e3o praticamente id\u00eanticas nos dois quadros "
            "(26\u201329% \u00d7 90\u201394% do quadro), o que descarta oclus\u00e3o e geometria do recorte como "
            "causa. <b>Causa raiz:</b> a U\u00b2-Net \u00e9 um detector de objeto <i>saliente</i> e n\u00e3o recebe "
            "a caixa do YOLO como condicionante \u2014 ela apenas processa o recorte. Nada obriga a "
            "produzir m\u00e1scara numa regi\u00e3o onde j\u00e1 se sabe que h\u00e1 uma pessoa, ent\u00e3o a etapa falha em "
            "sil\u00eancio. <b>O mesmo mecanismo em escala:</b> no v\u00eddeo <i>skate</i>, com multid\u00e3o, "
            "<b>68,9% das caixas (1092 de 1586) produziram m\u00e1scara vazia</b>, incluindo pessoas "
            "detectadas com confian\u00e7a 0,86 e caixa de 27 mil pixels. No quadro 112, cinco caixas "
            "entram na U-Net e apenas uma sai com m\u00e1scara: a do skatista em primeiro plano. Um "
            "segmentador condicionado pela caixa resolveria este modo de falha; baixar o limiar, "
            "n\u00e3o."
        ),
        "painel": "archery_f000291.png",
        "legenda": "Quadro 291: original \u00b7 caixas do YOLO \u00b7 m\u00e1scara da U-Net (quase vazia) \u00b7 "
                   "sa\u00edda integrada.",
    },
    {
        "titulo": "Erro 2 \u2014 O SlowFast confunde a fase de prepara\u00e7\u00e3o com <i>yoga</i> "
                  "(<i>pushups</i>, 0\u20132,1 s).",
        "texto": (
            "As quatro primeiras janelas classificam <i>yoga</i> (confian\u00e7a at\u00e9 0,93) antes de "
            "estabilizar em <i>push up</i>. A causa \u00e9 temporal, n\u00e3o espacial: a janela de 32 quadros "
            "cobre ~1,1 s e, nesse intervalo inicial, a pessoa est\u00e1 se abaixando e posicionando as "
            "m\u00e3os no ch\u00e3o \u2014 um movimento que de fato se parece mais com uma postura de ioga do "
            "que com uma flex\u00e3o. O top-5 cont\u00e9m <i>push up</i> em 95% das janelas, ou seja, o "
            "modelo n\u00e3o ignora a classe correta, apenas a ordena abaixo nessa fase. O efeito \u00e9 "
            "intr\u00ednseco \u00e0 janela deslizante: toda fronteira de a\u00e7\u00e3o produz uma janela que mistura "
            "dois movimentos. Janelas menores reduziriam a mistura, mas o SlowFast 8\u00d78 foi treinado "
            "com esse comprimento temporal e encurt\u00e1-lo degradaria as demais janelas."
        ),
        "painel": "pushups_f000048.png",
        "legenda": "Quadro 48 (1,6 s), janela em que o top-1 passa de <i>yoga</i> para <i>push up</i>.",
    },
    {
        "titulo": "Erro 3 \u2014 A multid\u00e3o fragmenta as identidades do rastreador (<i>skate</i>).",
        "texto": (
            "Com 4,4 pessoas por quadro (m\u00e1ximo de 10), o ByteTrack emite <b>124 <i>track_ids</i> "
            "distintos em 360 quadros</b> para um punhado de pessoas reais. A mediana de vida de um "
            "<i>track</i> \u00e9 de 5 quadros, e 66 dos 124 vivem 5 quadros ou menos: o rastreador perde "
            "e recria a mesma pessoa a cada oclus\u00e3o m\u00fatua. O efeito aparece nas demais m\u00e9tricas "
            "\u2014 a IoU temporal cai de 0,97\u20130,99 nos outros v\u00eddeos para 0,77, e a confian\u00e7a m\u00e9dia "
            "do YOLO cai de 0,81\u20130,93 para 0,63. <b>Causa:</b> o ByteTrack associa por IoU entre "
            "quadros consecutivos, e numa plateia densa as caixas de pessoas diferentes se "
            "sobrep\u00f5em mais entre si do que a caixa de uma pessoa se sobrep\u00f5e a ela mesma depois de "
            "um movimento r\u00e1pido. Vale separar os dois fen\u00f4menos: a <i>detec\u00e7\u00e3o</i> continua "
            "correta, com cobertura de 99,4% dos quadros. O que quebra \u00e9 a identidade, n\u00e3o a "
            "localiza\u00e7\u00e3o \u2014 e \u00e9 a identidade que o enunciado cobra ao perguntar se a caixa "
            "acompanha <i>a pessoa</i>."
        ),
        "painel": "skate_f000112.png",
        "legenda": "Quadro 112: cinco pessoas detectadas, uma \u00fanica m\u00e1scara produzida (ver erro 1).",
    },
]

EXTENSAO = (
    "O resultado depende de haver um sujeito dominante, e os tr\u00eas v\u00eddeos cobrem os tr\u00eas casos. "
    "Em <i>archery</i> as duas variantes empatam em 100% de acerto e 1,00 de confian\u00e7a: o sujeito "
    "j\u00e1 domina o quadro, ent\u00e3o recortar n\u00e3o muda nada. Em <i>pushups</i> o recorte da pessoa "
    "acerta 82% das janelas contra 36% do quadro inteiro, com confian\u00e7a m\u00e9dia de 0,81 contra 0,60, "
    "e as duas variantes concordam em apenas 36% das janelas \u2014 a pessoa deitada ocupa uma faixa "
    "estreita no centro de um quadro dominado por c\u00e9u e areia, e o <i>center-crop</i> de 256 px do "
    "pr\u00e9-processamento descarta justamente as extremidades onde o movimento acontece. Em "
    "<i>skate</i> o recorte <b>n\u00e3o ajuda</b> (68% contra 77% do quadro inteiro), e a raz\u00e3o \u00e9 "
    "estrutural: com 4,4 pessoas espalhadas pela cena, a uni\u00e3o das caixas cobre praticamente o "
    "quadro todo (601\u00d7359 de 640\u00d7360), de modo que a variante <i>pessoa</i> degenera na variante "
    "<i>quadro</i> e s\u00f3 acrescenta tremor de recorte entre janelas. <b>Conclus\u00e3o:</b> alimentar o "
    "SlowFast com o recorte da pessoa compensa quando h\u00e1 um sujeito \u00fanico e pequeno em rela\u00e7\u00e3o "
    "ao quadro; com multid\u00e3o, a uni\u00e3o das caixas perde o sentido e seria preciso recortar por "
    "<i>track</i>, um por vez."
)

CONCLUSAO = (
    "A integra\u00e7\u00e3o funciona e os tr\u00eas modelos pr\u00e9-treinados se sustentam sem nenhum ajuste de "
    "pesos. A etapa mais confi\u00e1vel \u00e9 a detec\u00e7\u00e3o, que mant\u00e9m cobertura acima de 99% nos tr\u00eas "
    "v\u00eddeos; a mais fr\u00e1gil \u00e9 a segmenta\u00e7\u00e3o, porque a U-Net do pipeline n\u00e3o \u00e9 condicionada pela "
    "caixa e pode falhar silenciosamente mesmo quando a detec\u00e7\u00e3o est\u00e1 correta \u2014 em cena com "
    "multid\u00e3o, em quase 70% das caixas. O SlowFast \u00e9 confi\u00e1vel enquanto a a\u00e7\u00e3o \u00e9 cont\u00ednua e "
    "identific\u00e1vel no vocabul\u00e1rio do Kinetics-400, e sua confian\u00e7a \u00e9 um indicador honesto: ela "
    "cai justamente nas fronteiras de a\u00e7\u00e3o, o que torna a pr\u00f3pria confian\u00e7a um sinal \u00fatil para "
    "segmentar o v\u00eddeo em trechos. Se fosse para estender o trabalho, a prioridade seria "
    "condicionar a m\u00e1scara \u00e0 detec\u00e7\u00e3o e recortar por <i>track</i> para o SlowFast \u2014 n\u00e3o trocar "
    "de classificador de a\u00e7\u00e3o."
)

CREDITOS = (
    "V\u00eddeos: <i>archery</i> \u2014 amostra de pesquisa do PyTorchVideo (clipe do Kinetics-400); "
    "<i>pushups</i> e <i>skate</i> \u2014 Wikimedia Commons, CC BY-SA 4.0. Proced\u00eancia completa em "
    "data/videos/LICENSES.md. C\u00f3digo, m\u00e9tricas e v\u00eddeos de demonstra\u00e7\u00e3o acompanham esta entrega."
)
