
# B5W7: Shipping a Data Product — Week 7 Challenge | 10 Academy

## 🗂 Challenge Context

This repository documents the submission for 10 Academy’s **B5W7: Shipping a Data Product** challenge.

Kara Solutions, a leading data science firm in Ethiopia, aims to analyze Telegram channels related to Ethiopian medical businesses. This project builds a production-grade ELT pipeline that scrapes unstructured Telegram data, enriches it with computer vision, and delivers structured insights through a dimensional data warehouse and an analytical API.

Key questions addressed:
- What are the most frequently mentioned medical products or drugs across Telegram?
- How does price or availability vary across channels?
- Which channels post the most visual content?
- What are the daily and weekly trends in health-related discussions?

This end-to-end pipeline is built using Telethon, dbt, YOLOv8, FastAPI, and Dagster.

---

## 🛠 Project Features

- 📥 **Data Ingestion**: Scraping public Telegram channels using the Telethon API
- 🗃 **Data Lake**: Raw JSON and images organized in a partitioned file system
- 🛠 **Dimensional Modeling**: dbt-based star schema built in PostgreSQL
- 🧼 **Transformation**: Multi-layered staging and data marts with dbt tests
- 🧠 **YOLOv8 Enrichment**: Detects objects in medical product images
- 🌐 **FastAPI Interface**: Exposes insights via custom analytical endpoints
- 📆 **Dagster Orchestration**: Schedules and monitors the full ELT pipeline

---

## 🔧 Project Setup

1. Clone the repository:

```bash
git clone https://github.com/NabloP/b5w7-shipping-a-data-product-challenge.git
cd b5w7-shipping-a-data-product-challenge
```

2. Create and activate the virtual environment:

**On Windows (PowerShell):**
```powershell
python -m venv data-product-challenge
.\data-product-challenge\Scripts\Activate
```

**On macOS/Linux:**
```bash
python3 -m venv data-product-challenge
source data-product-challenge/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🐳 Docker Setup (Optional for Cross-System Portability)

Use Docker Compose to spin up both the app and PostgreSQL:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Ensure secrets are placed in `docker/.env` and not committed to version control.

---

## 🔐 Business Use Case

### 1. Why This Pipeline?

Public Telegram channels serve as a rich but untapped source of market signals. For Ethiopia’s fragmented healthcare market, structured analysis of these channels can:
- Track drug availability and demand
- Identify reseller activity or stockouts
- Monitor price fluctuations and visual marketing trends

### 2. Pipeline Value Chain

- **Extract**: Get structured + image data from high-volume public channels
- **Load**: Store in raw partitioned folders and PostgreSQL raw schema
- **Transform**: Clean and enrich with YOLO + dbt models
- **Expose**: Deliver insights via secure APIs for business users

---

## 📁 Project Structure

<!-- TREE START -->

```
b5w7-shipping-a-data-product-challenge/
├── data/
│   └── raw/
│       ├── telegram_messages/
│       └── telegram_images/
├── dbt/
│   └── telegram_dbt/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env
├── notebooks/
├── src/
│   ├── scraping/
│   │   └── telegram_scraper.py
│   ├── ingestion/
│   │   └── json_to_postgres_loader.py
│   ├── utils/
│       ├── config.py
│       └── logger.py
├── requirements.txt
├── run_pipeline.sh
└── README.md
```
<!-- TREE END -->
---

## ✅ Task Tracker

| Task # | Task Name                            | Status       | Description |
|--------|--------------------------------------|--------------|-------------|
| 0      | Environment Setup & Docker           | ✅ Completed | Virtual env, Docker, env var management |
| 1      | Telegram Scraping                    | 🚧 In Progress | Scraper for messages + images with logging |
| 2      | Data Modeling with dbt               | 🚧 In Progress | Raw → staging → mart with dbt & Postgres |
| 3      | YOLOv8 Image Enrichment              | ⏳ Pending    | Detect object types in scraped images |
| 4      | FastAPI Analytical Interface         | ⏳ Pending    | API endpoints for search + product metrics |
| 5      | Dagster Pipeline Orchestration       | ⏳ Pending    | Job ops, scheduling, Dagster UI |

---

## 📚 References

- [Telethon](https://docs.telethon.dev)
- [dbt](https://docs.getdbt.com)
- [YOLOv8](https://docs.ultralytics.com)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Dagster](https://docs.dagster.io)

---

## 👤 Author

**Nabil Mohamed**  
10 Academy AIM Bootcamp Participant  
GitHub: [@NabloP](https://github.com/NabloP)