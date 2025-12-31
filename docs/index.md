---
layout: home
title: Home
nav_order: 1
---

# 🚀 QualiDB AI
{: .fs-9 }

**Your AI-powered Data Governance Assistant.**
{: .fs-6 .fw-300 }

QualiDB is a CLI tool that automates database documentation using LLMs (Claude, Gemini, GPT) and runs automated data quality checks (Freshness, Nulls, Uniqueness).

[Get Started](./usage.html){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View Source](https://github.com/YOUR_USERNAME/QualiDB){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## 🔥 Key Features

- **🧠 AI Documentation:** Auto-generate business descriptions for tables and columns using Claude 3.5 Sonnet or Gemini.
- **🔍 Data Quality Inspector:** Run instant checks for Nulls, Duplicates, and Freshness.
- **💾 Database Sync:** Push approved documentation back into your database (BigQuery, Snowflake, Postgres) as comments.
- **🛡️ BigQuery Native:** Smart handling of `project.dataset.table` schemas and strict quoting.

## ⚡ Quick Install

```bash
# Clone the repo
git clone [https://github.com/YOUR_USERNAME/QualiDB.git](https://github.com/YOUR_USERNAME/QualiDB.git)
cd QualiDB

# Install dependencies
pip install -r requirements.txt

# Run the tool
python db_check.py