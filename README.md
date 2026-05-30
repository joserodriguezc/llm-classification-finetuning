# LLM Classification Finetuning

> **Kaggle Competition** · [Ver en Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning)

Predicción de preferencias humanas en batallas cara a cara entre LLMs, usando conversaciones reales de [Chatbot Arena](https://chat.lmsys.org/).

---

## 📌 Descripción del Problema

Dado un prompt y las respuestas de dos modelos de lenguaje (A y B), el objetivo es predecir cuál respuesta prefiere el usuario — o si hay empate.

| Campo | Descripción |
|---|---|
| `prompt` | Instrucción o pregunta enviada a ambos modelos |
| `response_a / response_b` | Respuestas generadas por cada modelo |
| `winner_model_a / winner_model_b / tie` | Variable target (3 clases) |

**Métrica de evaluación:** Log Loss

---

## 🧪 Enfoque y Metodología

El proyecto está estructurado en tres notebooks progresivos:

### `01_eda.ipynb` — Análisis Exploratorio
- Distribución de clases (balance del dataset)
- Análisis de longitud de prompts y respuestas
- Identificación de patrones de preferencia
- Visualización de sesgos conocidos (verbosidad, posición)

### `02_baseline.ipynb` — Modelo de Referencia
- Feature engineering sobre texto (longitud, ratio, diferencia de tokens)
- TF-IDF + Logistic Regression / XGBoost
- Establecimiento de score base en el leaderboard

### `03_deberta_finetuning.ipynb` — Fine-tuning con Transformers
- Fine-tuning de `DeBERTa-v3-base` con HuggingFace Transformers
- Concatenación de prompt + response_a + response_b como input
- Entrenado en Kaggle Notebooks (GPU T4)

---

## 📊 Resultados

| Modelo | Log Loss (Public LB) |
|---|---|
| Baseline TF-IDF + LogReg | — |
| DeBERTa-v3-base fine-tuned | — |

> *Tabla en construcción — scores se actualizan con cada submission.*

---

## 🛠️ Stack Técnico

- **Python 3.11** · `uv` para gestión de dependencias
- **HuggingFace Transformers** — fine-tuning de DeBERTa
- **PyTorch** — backend de entrenamiento
- **scikit-learn / XGBoost** — modelos de baseline
- **pandas / numpy** — procesamiento de datos
- **Kaggle Notebooks** — entorno de entrenamiento GPU (T4)

---

## 🔁 Reproducibilidad

Los notebooks fueron desarrollados y entrenados en **Kaggle Notebooks** con GPU habilitada.

Para reproducir localmente el entorno de análisis:

```bash
git clone https://github.com/joserodriguezc/llm-classification-finetuning.git
cd llm-classification-finetuning

uv venv --python 3.11
.venv\Scripts\activate.bat   # Windows CMD
uv sync
```

Los datos deben descargarse desde la página oficial de la competencia en Kaggle y ubicarse en `data/raw/`.

---

## 📁 Estructura del Repositorio

```
llm-classification-finetuning/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_deberta_finetuning.ipynb
│
├── src/
│   └── utils.py
│
├── submissions/
│   └── (archivos de submission — no versionados)
│
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

---

## 👤 Autor

**José Ignacio Rodríguez**
Especialista en datos con experiencia en gobierno de datos, analytics e IA aplicada en educación superior.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/joigrodriguez/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/joignacio)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/joserodriguezc)