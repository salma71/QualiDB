---
layout: default
title: CLI Reference
nav_order: 4
---

# ⚙️ CLI Reference
{: .no_toc }

QualiDB is designed for interactivity, but it also supports robust command-line flags. These options allow you to automate workflows, integrate with CI/CD pipelines, or simply bypass the interactive menus.

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Global Options

These flags can be used with any command to configure the runtime environment.

| Flag | Description | Example |
| :--- | :--- | :--- |
| `--conn` | **Connection String:** Pass a SQLAlchemy connection string directly to bypass the interactive database selector. | `bigquery://project_id:dataset_id` |
| `--help` | **Help:** Show the help message and exit. | `python db_check.py --help` |

---

## The Checks

### 🕒 Freshness
- **Logic:** Automatically detects `TIMESTAMP` and `DATE` columns. Calculates the time difference between `NOW()` and the maximum value in that column.
- **Alert:** Flags data older than 24 hours as **⚠️ OLD**.

### 🚫 Null Values
- **Logic:** Scans columns for missing data (`NULL`).
- **Alert:** Returns **❌ FAIL** if any nulls are found in critical columns.

### 🆔 Uniqueness
- **Logic:** Checks Primary Keys or ID columns for duplicate entries.
- **Alert:** Returns **❌ FAIL** if `COUNT(DISTINCT col) != COUNT(col)`.

## Automatic Reporting
Results are displayed in a high-contrast ASCII table directly in your terminal, making it easy to spot failures at a glance.

## Commands

### 1. `generate-schema`
**Description:** Triggers the AI documentation workflow. It scans your schema and uses an LLM to generate business-friendly descriptions.

**Usage:**
```bash
python db_check.py generate-schema [OPTIONS]
--conn      The database connection string (prompts if not provided).
--output    Path to output file. (Default: models/schema.yml)
--model     AI Model to use. (Default: gpt-3.5-turbo)
--write-db  Write descriptions directly to the database. (Flag)
--help      Show this message and exit.
```


### 2. `check-quality`
**Description:** Runs the automated data quality inspector. It checks for freshness, null values, and uniqueness constraints.

**Usage:**
```bash
python db_check.py check-quality [OPTIONS]
--conn      The database connection string. (Env: DB_CONNECTION_STRING)
--help      Show this message and exit.
```

### 3. `push-to-db`
**Description:** Reads your local documentation file (schema.yml) and pushes the descriptions back to the database using COMMENT ON or ALTER TABLE statements.

**Usage:**
```bash
python db_check.py push-to-db [OPTIONS]
--input     Path to the YAML file containing the descriptions. (Default: models/schema.yml)
--conn      The database connection string. (Env: DB_CONNECTION_STRING)
--help      Show this message and exit.
```

**Example:**
```bash
python db_check.py push-to-db --conn "bigquery://project_id:dataset_id"
```

