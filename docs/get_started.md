---
layout: default
title: Getting Started
nav_order: 2
---

# 🚀 Getting Started with QualiDB
{: .no_toc }

QualiDB is an all-in-one Data Governance assistant. It uses AI to document your database and automated checks to ensure data quality.

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 🛠️ Prerequisites

Before running the tool, ensure you have:
* **Python 3.10+** installed.
* Access to a database (BigQuery, PostgreSQL, Snowflake, or SQLite).
* An API Key for your preferred AI provider:
    * `ANTHROPIC_API_KEY` (Claude - *Recommended*)
    * `GOOGLE_API_KEY` (Gemini)
    * `OPENAI_API_KEY` (GPT-4)

---

## 📥 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/QualiDB.git](https://github.com/YOUR_USERNAME/QualiDB.git)
    cd QualiDB
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your keys:
    ```bash
    # AI Provider Keys (Only one is required)
    ANTHROPIC_API_KEY="sk-ant-..."
    GOOGLE_API_KEY="AIza..."
    OPENAI_API_KEY="sk-..."

    # Database Connection (Optional, can be passed via CLI)