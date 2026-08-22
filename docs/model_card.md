# Model Card — Churn Guard MLP

> Documento vivo. Atualizar a cada retreino que for promovido a produção.

| Campo | Valor |
|---|---|
| **Nome** | Churn Guard MLP |
| **Versão do modelo** | 1.0.0 |
| **Versão do dataset** | v1.0 |
| **Tipo** | Classificação binária (churn / não churn) |
| **Arquitetura** | MLP densa em PyTorch — `[44 → 128 → 64 → 32 → 1]`, BatchNorm + ReLU + Dropout(0.3) |
| **Parâmetros treináveis** | 16.577 |
| **Framework** | PyTorch 2.x |
| **Data de treino** | 22/08/2026 |
| **Responsáveis** | Matheus Angelo Filetti (rm373015) |
| **Licença** | Uso acadêmico — POSTECH / FIAP, Machine Learning Engineering |

---

## 1. Uso pretendido

**Caso de uso primário.** Priorizar a fila de contato ativo do time de retenção. O modelo pontua a base de clientes e devolve uma probabilidade de cancelamento, traduzida em três faixas de risco. O time trabalha a faixa alta primeiro.

**Usuários.** Analistas de retenção (via export para o CRM) e sistemas internos que consomem a API `/predict`.

**Decisão apoiada.** *Quem contatar primeiro* — não *a quem negar serviço*. O modelo é um instrumento de priorização, não de elegibilidade.

### Usos fora de escopo

- Negar, encarecer ou restringir serviço a um cliente com base no score.
- Decisão automatizada sem revisão humana. Toda ação derivada do score passa por um analista.
- Precificação individual ou diferenciação de tarifa por score.
- Generalização para outra operadora, outro país ou outro produto sem revalidação completa. O dataset é de uma operadora norte-americana específica.
- Inferência causal. O modelo aprende correlações; não indica que mudar um atributo causa retenção.

---

## 2. Dados

**Origem.** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7.043 clientes, 21 colunas. Dataset público, distribuído pela IBM para fins educacionais.

**Alvo.** `Churn` — cliente cancelou no último mês. Distribuição observada: **26,5% positivos**.

**Features utilizadas (18).**

| Tipo | Colunas |
|---|---|
| Numéricas (3) | `tenure`, `MonthlyCharges`, `SeniorCitizen` |
| Categóricas (15) | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

**Features removidas e por quê.**

- `customerID` — identificador, sem poder preditivo e com risco de memorização.
- `TotalCharges` — correlação de ~0,83 com `tenure` (é essencialmente `tenure × MonthlyCharges`). Além disso vem quebrada no CSV: os 11 clientes com `tenure=0` têm string vazia no lugar do valor. Manter as duas introduz colinearidade sem ganho.

**Pré-processamento.**

- Numéricas: `SimpleImputer(median)` → `StandardScaler`
- Categóricas: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- Resultado: **44 features densas**.
- O `ColumnTransformer` é ajustado **exclusivamente** em treino; validação e teste recebem só `transform`.

**Split.** Estratificado por `Churn`, `SEED=42`:

| Conjunto | n | % | Taxa de churn |
|---|---|---|---|
| Treino | 4.929 | 70,0% | 26,5% |
| Validação | 1.057 | 15,0% | 26,6% |
| Teste | 1.057 | 15,0% | 26,5% |

**Validação de contrato.** `pandera` valida tipos, intervalos, unicidade de ID e vocabulário categórico na ingestão. Schema quebrado interrompe o treino (`src/churn/data/validate.py`).

---

## 3. Treinamento

| Hiperparâmetro | Valor |
|---|---|
| Camadas ocultas | `[128, 64, 32]` |
| Dropout | 0.3 |
| Otimizador | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| Loss | `BCEWithLogitsLoss` com `pos_weight = n_neg/n_pos = 2.768` |
| Batch size | 64 |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, monitorando `val_auc`) |
| Gradient clipping | `max_norm=5.0` |
| Épocas máximas | 200 |
| **Early stopping** | `patience=15`, `min_delta=1e-4`, monitorando `val_auc`, com `restore_best_weights=True` |

**Sobre o early stopping.** O treino para quando o AUC de validação não melhora por 15 épocas consecutivas. Os pesos da melhor época ficam guardados em memória e são restaurados ao final — o modelo servido é o da melhor época de validação, **nunca o da última**. A época escolhida é registrada em `best_epoch` nos metadados e no MLflow.

**Execução registrada (22/08/2026).** Early stopping disparou na **época 36**, restaurando os pesos da **época 21** (melhor `val_auc` = 0.8299). Treino concluído em ~14s em Apple Silicon (device MPS). Três execuções consecutivas produziram métricas idênticas até a sexta casa decimal, confirmando a reprodutibilidade com `SEED=42`.

**Desbalanceamento.** Tratado por `pos_weight` na loss, equivalente ao `class_weight='balanced'` dos baselines. Optamos por reponderação em vez de SMOTE: oversampling sintético em espaço one-hot gera combinações categóricas que não existem no mundo real (por exemplo, cliente com `InternetService=No` e `StreamingTV=Yes`).

**Threshold de decisão.** Não é 0.5. É buscado por varredura no conjunto de **validação**, maximizando F1 sujeito a `recall ≥ 0.83`. O valor selecionado foi **0.43**, gravado em `models/mlp_metadata.json` e aplicado pela API.

---

## 4. Métricas

Avaliação no conjunto de **teste** (1.057 clientes, nunca vistos em treino nem na escolha de threshold).

| Modelo | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|
| DummyClassifier | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Regressão Logística | 0.8467 | 0.7964 | 0.6169 | 0.5034 |
| LightGBM (Optuna) | **0.8525** | 0.8107 | **0.6421** | 0.5316 |
| **MLP (PyTorch)** | 0.8444 | **0.8714** | 0.6070 | 0.4656 |

**Métrica primária: Recall.** O custo assimétrico manda. Um falso negativo é um cliente que cancela sem nunca ter recebido oferta de retenção — perde-se o LTV inteiro. Um falso positivo é um desconto oferecido a quem ficaria de qualquer forma — custa o valor do desconto. Como o LTV supera o desconto em uma ordem de grandeza, aceitamos precisão menor em troca de recall alto.

**Gates de promoção.** AUC-ROC ≥ 0.85, Recall ≥ 0.83, F1 ≥ 0.65, latência p95 ≤ 200ms. Verificados automaticamente em `quality_gate()`; o resultado vai para as tags do MLflow.

**Resultado dos gates.** Recall aprovado (0.8714 ≥ 0.83) e latência aprovada (~7ms ≤ 200ms). AUC-ROC (0.8444) e F1 (0.6070) ficaram abaixo das metas de 0.85 e 0.65.

A MLP lidera na métrica primária declarada: captura 87,1% dos clientes que cancelam, contra 81,1% do LightGBM — 6 pontos percentuais a mais de churners identificados. O custo é precisão menor (0.4656 vs 0.5316), consequência direta do threshold de 0.43, escolhido justamente para privilegiar recall. A diferença de AUC entre os dois modelos (0.8444 vs 0.8525) está dentro do ruído esperado para um teste com 1.057 amostras.

**Leitura honesta do resultado.** Gradient boosting continua sendo referência forte em dados tabulares desse porte, e o LightGBM vence em AUC e F1. A MLP vence onde o projeto declarou que importa. Nenhum dos dois atinge todas as metas — elas foram fixadas antes do treino e mantidas sem afrouxamento.

---

## 5. Análise de fairness

**Atributos auditados.** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
Avaliação no conjunto de teste (n=1.057), threshold 0.43.
Reproduzível com `python -m churn.models.fairness_audit`.

### Métricas por grupo

| Atributo | Grupo | n | Taxa churn real | Taxa seleção | Recall | FNR |
|---|---|---|---|---|---|---|
| `gender` | Female | 529 | 0.270 | 0.505 | 0.874 | 0.126 |
| `gender` | Male | 528 | 0.259 | 0.487 | 0.869 | 0.131 |
| `SeniorCitizen` | 0 | 888 | 0.231 | 0.440 | 0.829 | 0.171 |
| `SeniorCitizen` | 1 | 169 | 0.444 | 0.787 | 0.987 | 0.013 |
| `Partner` | No | 556 | 0.333 | 0.612 | 0.897 | 0.103 |
| `Partner` | Yes | 501 | 0.190 | 0.367 | 0.821 | 0.179 |
| `Dependents` | No | 724 | 0.314 | 0.599 | 0.903 | 0.097 |
| `Dependents` | Yes | 333 | 0.159 | 0.270 | 0.736 | 0.264 |

### Gate de fairness

| Atributo | FNR mín. | FNR máx. | Disparidade | Situação |
|---|---|---|---|---|
| `gender` | 0.126 | 0.131 | 0.006 | aprovado |
| `Partner` | 0.103 | 0.179 | 0.076 | aprovado |
| `SeniorCitizen` | 0.013 | 0.171 | 0.157 | **reprovado** |
| `Dependents` | 0.097 | 0.264 | 0.167 | **reprovado** |

Limite adotado: disparidade de FNR ≤ 0.10. **Resultado: reprovado** em `SeniorCitizen` e `Dependents`.

**Por que FNR e não paridade demográfica.** Um falso negativo significa cliente em risco que não recebeu ação de retenção. Disparidade sistemática de FNR entre grupos indica que um grupo é sub-atendido pelo programa. Igualar taxas de seleção entre grupos com taxas de churn genuinamente diferentes prejudicaria o grupo de maior risco — por isso paridade demográfica seria a métrica errada aqui.

### Interpretação

As duas violações seguem o mesmo padrão: o grupo com FNR mais alto é sempre o de **menor taxa de churn de base**. `Dependents=Yes` churna 15,9% contra 31,4% de `Dependents=No`; `SeniorCitizen=0` churna 23,1% contra 44,4% dos idosos.

Isso é consequência mecânica de um threshold global único. A distribuição de scores se desloca junto com a taxa base, de modo que o mesmo corte (0.43) captura proporções distintas em cada grupo.

Não é evidência de que o modelo penalize um grupo protegido. No caso de `SeniorCitizen`, o grupo minoritário é o **super-selecionado**: taxa de seleção de 78,7%, FNR de 0,013 e FPR de 0,628 — o modelo praticamente sinaliza todo idoso, e portanto esse grupo recebe *mais* ação de retenção, não menos.

Como a ação derivada do score é uma oferta de retenção e nunca uma restrição de serviço, o prejuízo de estar do lado de FNR alto é receber menos contato preventivo — e o grupo nessa posição é justamente o de menor risco real de cancelamento.

`gender`, o atributo protegido de maior sensibilidade jurídica, apresenta a menor disparidade da auditoria (0.006).

### Mitigações consideradas

1. **Thresholds por grupo** — equalizaria o FNR, mas implica tratamento diferenciado explícito por atributo protegido, o que levanta questão jurídica própria. Não adotado nesta fase.
2. **Reponderação por subgrupo no treino** — via `pos_weight` estratificado. Candidato à Fase 2.
3. **Manter o gate reprovado e documentado** — **adotado**. O gate cumpriu seu papel ao detectar a disparidade; ocultá-la ou afrouxar o limite para 0.20 apenas para obter aprovação anularia o mecanismo.

### Limitações

- `SeniorCitizen=1` tem n=169 (IC ≈ ±7 p.p.). Trate a disparidade nesse recorte como indicativa, não conclusiva.
- `gender` é binário no dataset; clientes não-binários não são representados e o modelo não tem como servi-los adequadamente.
- Sem dados de raça, renda ou geografia — não podemos auditar essas dimensões, o que não significa que não haja disparidade nelas.
- Fairness foi **medida, não otimizada**. Não houve mitigação in-processing nem pós-processamento.

---

## 6. Limitações e riscos

**Do modelo.**

- Dataset estático, de recorte único. Não há componente temporal; o modelo não captura sazonalidade nem tendência de mercado.
- 7.043 registros é pequeno para uma rede neural. A vantagem da MLP sobre gradient boosting em dados tabulares desse porte é marginal — e os resultados da seção 4 confirmam isso. A escolha é pedagógica e explicitamente assumida.
- Sem features de comportamento (uso de dados, chamadas ao SAC, histórico de reclamações), que na prática são os sinais mais fortes de churn.
- Correlação, não causalidade. `Contract=Month-to-month` prediz churn; forçar migração de contrato não necessariamente retém.
- Precisão de 0.4656 significa que ~53% dos clientes sinalizados não cancelariam. Dimensione o custo da campanha considerando isso.

**De uso.**

- **Profecia autorrealizável.** Se o time só contata a faixa alta, o resultado observado passa a depender da ação. Manter um grupo de controle sem contato para medir o efeito real.
- **Feedback loop.** Retreinar com dados pós-intervenção sem marcar quem recebeu ação contamina o rótulo. Registre a intervenção como feature ou exclua os tratados.
- **Drift.** Mudança de portfólio, preço ou concorrência invalida a distribuição de treino. Ver `docs/monitoring_plan.md`.

---

## 7. Manutenção

| Item | Definição |
|---|---|
| Retreino programado | Trimestral |
| Retreino disparado | Drift em qualquer gate do plano de monitoramento |
| Owner | Time de Advanced Analytics |
| Rastreamento | MLflow — experimento `churn-prediction-fase1` |
| Rollback | Versão anterior no MLflow Model Registry; `/health` reporta `degraded` se os artefatos não carregarem |

**Contato para questionamento de decisão.** Cliente ou analista que discorde de uma priorização deve escalar ao owner do modelo. Como o score não nega serviço, não há decisão automatizada adversa a contestar — mas o registro do questionamento alimenta a revisão trimestral.

---

## 8. Referências

- Mitchell et al. (2019), *Model Cards for Model Reporting*
- [Fairlearn](https://fairlearn.org/) — metodologia de métricas por grupo
- [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)