# RAG Evaluation

## Projektbeskrivning

Detta projekt är en experimentell studie av en RAG-baserad (Retrieval-Augmented Generation) AI-lösning.

Syftet är att undersöka hur olika chunk-storlekar påverkar:

- kvaliteten på informationshämtningen (retrieval quality)
- korrektheten i de genererade svaren (answer correctness)

RAG-systemet använder 10 dokument om länder i södra Afrika som kunskapskälla. Dokumenten delas upp i mindre textdelar, så kallade chunks, som omvandlas till embeddings. När en fråga ställs används cosine similarity för att hitta de tre mest relevanta chunkarna. Dessa skickas sedan tillsammans med frågan till en språkmodell som genererar svaret.

Tre olika chunk-storlekar testas:

- 500 tecken
- 1000 tecken
- 1500 tecken

Övriga huvudsakliga delar av systemet hålls konstanta under experimentet.

---

## Teknisk uppbyggnad

RAG-pipelinen fungerar i följande steg:

```text
Dokument
   ↓
Dela upp i chunks
   ↓
Skapa embeddings
   ↓
Behåll chunks + embeddings
   ↓
Användarfråga
   ↓
Skapa embedding för frågan
   ↓
Cosine similarity
   ↓
Hämta top 3 chunks
   ↓
Fråga + hämtad text
   ↓
LLM
   ↓
Genererat svar

```

## Installation
### 1. Klona projektet från GitHub

git clone https://github.com/emanuelssonlinnea-rgb/RAG_evaluation_examinerande_uppgift.git

### 2. Skapa en virtuell miljö

    python -m venv venv

Windows PowerShell 

    venv\Scripts\Activate.ps1

macOS/Linux 

    source venv/bin/activate

### 3. Installera beroenden

    python -m pip install -r requirements.txt

#### Projektet använder:

- Python
- OpenAI API
- text-embedding-3-small för embeddings
- GPT-5.6 Luna för att generera och utvärdera svar
- pandas för datahantering och analys
- matplotlib för visualisering
- cosine similarity för retrieval

Projektet använder ingen separat vektordatabas. Chunks och embeddings hanteras i Python under körningen.

### 4. Skapa en .env-fil

Projektet använder OpenAI API och behöver därför en API-nyckel.

Skapa en fil som heter:

    .env

i projektets rotmapp.

Lägg in:

    OPENAI_API_KEY=din-openai-api-nyckel

**Lägg aldrig upp din riktiga API-nyckel på GitHub!!**

## Starta och testa RAG-systemet
Testa systemet med exempel

Projektet innehåller run_demo.py, som används för att testa RAG-systemet med tre exempel:

- Single-passage-fråga
- Multi-passage-fråga
- No-answer-fråga

Kör:

    python run_demo.py

Programmet visar bland annat:

- frågan
- det genererade svaret
- de chunks som hämtades
- similarity score för varje chunk

Detta kan användas för att manuellt kontrollera hur RAG-systemet fungerar.

## Köra utvärderingen

Utvärderingen körs med test_evaluation.py.

Programmet använder frågorna i:

    data/all_questions.csv

och testar både retrieval och genererade svar.

Testa chunk size 500
    python test_evaluation.py --chunk-size 500

Resultatet sparas som:

    data/evaluation_results_500.csv

Testa chunk size 1000
    python test_evaluation.py --chunk-size 1000

Resultatet sparas som:

    data/evaluation_results_1000.csv

Testa chunk size 1500
    python test_evaluation.py --chunk-size 1500

Resultatet sparas som:

    data/evaluation_results_1500.csv

## Analysera resultaten

När de tre experimenten har körts kan resultaten analyseras med:

    python analyze_results.py

## Projektstruktur
```text
RAG_evaluation/
│
├── data/
│   ├── documents.csv
│   ├── all_questions.csv
│   ├── single_passage_questions.csv
│   ├── multi_passage_questions.csv
│   ├── no_answer_questions.csv
│   ├── evaluation_results_500.csv
│   ├── evaluation_results_1000.csv
│   └── evaluation_results_1500.csv
│
├── evaluation/
│   └── verification_log.csv
│
├── results/
│   ├── answer_correctness_by_chunk_size.png
│   ├── answer_correctness_by_question_type.png
│   ├── recall_by_chunk_size.png
│   └── recall_by_question_type.png
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── embeddings_manager.py
│   ├── evaluation.py
│   ├── rag_system.py
│   ├── retrieval_system.py
│   └── text_processor.py
│
├── run_demo.py
├── test_evaluation.py
├── analyze_results.py
├── README.md
├── report.md
├── requirements.txt
└── .gitignore
```