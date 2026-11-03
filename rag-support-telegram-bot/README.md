# RAG support Telegram bot

A technical-support assistant for an equipment manufacturer. It answers
customers' questions in Telegram with retrieval-augmented generation over product
manuals. Claude writes the answers, OpenAI embeddings power the search, and
PostgreSQL with pgvector stores both the document vectors and the conversations.

*Independent prototype, not the client system.*

## Task

Build a chatbot that cuts the load on a technical-support team:

- answer customer questions about the company's equipment from its manuals
  (PDF / DOC / DOCX);
- keep the dialogue context, limit the number of questions per user per day;
- let an administrator edit answers and send them to customers.

No data is included in this repository. [sample_data/](sample_data/) holds
synthetic manuals for made-up products, so you can run it.

## Solution

1. **Conversion** ([src/rag_bot/convert.py](src/rag_bot/convert.py)): LlamaParse turns
   PDF / DOC / DOCX manuals into markdown. Only new or changed files are parsed
   (`--force` converts all of them).
2. **Indexing** ([src/rag_bot/indexing.py](src/rag_bot/indexing.py),
   [src/rag_bot/store.py](src/rag_bot/store.py)): documents are split into overlapping chunks
   (512 tokens), embedded and stored in PostgreSQL. The `chunks` table has a pgvector
   column with an HNSW index.
   - Indexing is incremental: a document whose content hash has not changed is skipped,
     a changed one is replaced in one transaction, and a removed file is deleted from the index.
   - The embedding model and vector size are recorded, so vectors of different models
     never mix. Changing the model needs `--rebuild`.
3. **Retrieval** ([src/rag_bot/retrieval.py](src/rag_bot/retrieval.py)): one SQL query returns
   the top-k chunks (k = 8) by cosine similarity, with an optional similarity threshold.
   Each chunk is tagged with its source file. The bot reads the database directly, so
   re-indexing takes effect without a restart.
4. **Answer** ([src/rag_bot/answer.py](src/rag_bot/answer.py)):
   - the rules go into the system prompt ([prompts/system.txt](prompts/system.txt));
   - the user's questions and answers from the last 30 minutes are sent as earlier turns
     of the conversation;
   - the retrieved chunks go into the last message ([prompts/question.txt](prompts/question.txt));
   - the reply ends with the list of source documents.
5. **Conversation log** ([src/rag_bot/history.py](src/rag_bot/history.py)): a table in the
   same database, used through a shared connection pool ([src/rag_bot/db.py](src/rag_bot/db.py)).
   - The daily quota check and the insert of the question run in one transaction under a
     per-user lock. Parallel messages cannot exceed the quota, and a failed answer does
     not use it up.
   - `purge` deletes old conversations, so a data-retention period can be enforced.
6. **Telegram bot** ([src/rag_bot/bot.py](src/rag_bot/bot.py)): python-telegram-bot handlers.
   - Updates are processed concurrently. Quota, retrieval and the LLM call run in worker
     threads (at most 8 at a time), so one slow answer does not block other users.
   - Questions from the same user are answered one at a time, so answers arrive in order
     and a follow-up sees the previous answer.
   - Only new text messages are answered (not edits or channel posts).
   - Long answers are split to fit Telegram's 4096-unit limit, counted in UTF-16 as Telegram does.
   - Internal errors are logged, and users get a generic message.
7. **Administration** ([src/rag_bot/admin.py](src/rag_bot/admin.py)): list recent conversations,
   queue an edited reply for a message, send queued replies, purge old data. It can run
   while the bot is running. Sending is throttled and waits out Telegram flood-control errors.
8. **Evaluation** ([src/rag_bot/evaluate.py](src/rag_bot/evaluate.py)): retrieval hit rate
   and MRR for a set of questions with known source documents, at several similarity
   thresholds. Use it to tune `top_k` and `similarity_threshold`.

Providers are pluggable ([src/rag_bot/providers.py](src/rag_bot/providers.py)). With
`"offline"` the app uses a hashing bag-of-words embedding and a stand-in LLM that
returns the best retrieved chunk, so the pipeline runs without API keys. It still needs
PostgreSQL with pgvector.

## Results

Retrieval on the sample data ([sample_data/eval.json](sample_data/eval.json): 14 questions,
worded differently from the manuals), offline hashing embedding, chunks of 80 tokens, top_k = 3:

| Similarity threshold | Hit rate | MRR |
|---|---|---|
| 0.0 | 0.93 | 0.63 |
| 0.2 | 0.93 | 0.63 |
| 0.3 | 0.14 | 0.10 |

These numbers only show how the evaluation works: the offline embedding matches words,
not meaning, and the chunk size was chosen on these same 14 questions. Run the evaluation
with real embeddings and your own questions before you choose the production settings.

## Project structure

```
rag-support-telegram-bot/
├── README.md
├── requirements.txt
├── .env.example                 # API keys, bot token, DATABASE_URL (placeholders)
├── config/
│   └── settings.json            # production settings (Anthropic + OpenAI)
├── prompts/
│   ├── system.txt               # assistant rules
│   └── question.txt             # retrieved documentation + question
├── sample_data/                 # synthetic manuals, evaluation questions, offline settings
├── src/rag_bot/
│   ├── settings.py              # settings dataclass loaded from JSON
│   ├── providers.py             # Claude / OpenAI embeddings / offline stand-ins
│   ├── db.py                    # PostgreSQL connection pool (pgvector types)
│   ├── convert.py               # manuals -> markdown (LlamaParse)
│   ├── store.py                 # documents and chunk vectors in PostgreSQL
│   ├── indexing.py              # incremental indexing
│   ├── retrieval.py             # search
│   ├── answer.py                # prompt assembly, LLM call, sources
│   ├── history.py               # PostgreSQL conversation log, quota
│   ├── bot.py                   # Telegram bot
│   ├── admin.py                 # admin replies, data retention
│   └── evaluate.py              # retrieval evaluation
└── tests/
```

## Usage

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You need PostgreSQL with the pgvector extension, for example the official image:

```sh
docker run -d --name ragbot-db -e POSTGRES_PASSWORD=<password> -p 5432:5432 pgvector/pgvector:pg16
export DATABASE_URL=postgresql://postgres:<password>@localhost:5432/postgres
```

The tables are created on first use. The database role must be allowed to run
`CREATE EXTENSION vector`, or the extension must already be installed.

Try it offline on the sample data (no API keys needed):

```sh
cd src
python -m rag_bot.indexing --settings ../sample_data/settings.json
python -m rag_bot.answer   --settings ../sample_data/settings.json "What is the maximum cable length for the DX-10?"
python -m rag_bot.evaluate --settings ../sample_data/settings.json --cases ../sample_data/eval.json
```

Run with real models: copy `.env.example` to `.env` and fill it in. Put the manuals in
`data/source/`, where [config/settings.json](config/settings.json) points. Use a separate
database from the sample run, or index with `--rebuild`, because the embedding model differs.

```sh
cd src
python -m rag_bot.convert  --settings ../config/settings.json
python -m rag_bot.indexing --settings ../config/settings.json   # re-run after changing documents
python -m rag_bot.bot      --settings ../config/settings.json

python -m rag_bot.admin list
python -m rag_bot.admin reply 42 "Corrected answer"
python -m rag_bot.admin send
python -m rag_bot.admin purge --days 90
```

Tests run offline. Tests that use the index or the conversation log need a disposable
PostgreSQL database with pgvector; their tables are dropped and recreated. They are
skipped without it:

```sh
TEST_DATABASE_URL=postgresql://<user>:<password>@localhost:5432/<test-database> python -m pytest tests
```

The conversation log holds Telegram user IDs and their questions. Set a retention
period and run `admin purge` on a schedule.
