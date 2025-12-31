---
layout: default
title: Sample Output
nav_order: 4
---

# 📄 Sample Output
{: .no_toc }

See examples of what QualiDB generates for your project.

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 1. Generated Documentation (`schema.yml`)

When you run the **Generate Documentation** command, QualiDB uses AI to write descriptions and saves them to `models/schema.yml`.

**Terminal Output:**
```text
~/Desktop/QualiDB via 🐍 v3.11.6 (.venv)
> python db_check.py

? Select tables: (Use arrow keys to move, <space> to select, <a> to toggle, <i> to invert)
  ○ (Exit)
  ○ shipstation_raw.carriers
  ○ shipstation_raw.fulfillments
  ○ shipstation_raw.marketplaces
  ○ shipstation_raw.orders
  ○ shipstation_raw.products
» ● shipstation_raw.shipments
  ○ shipstation_raw.stores
  ○ shipstation_raw.users
  ○ shipstation_raw.warehouses
```

**Quality Check Results:**
```text
Scanning 2 tables in 'shipstation_raw'... 🏃
Scanning  [####################################]  100%

┌───────────┬──────────────┬───────────┬──────────────────┬──────────┐
│ Table     │ Column       │ Check     │ Result           │ Status   │
├───────────┼──────────────┼───────────┼──────────────────┼──────────┤
│ shipments │ shipment_id  │ Unique    │ PASS             │ ✅       │
│ shipments │ ship_date    │ Freshness │ 4.2h ago         │ ✅       │
│ shipments │ tracking_no  │ Nulls     │ FAIL (5 nulls)   │ ❌       │
└───────────┴──────────────┴───────────┴──────────────────┴──────────┘
```
**Command:**
```bash
python db_check.py generate-schema
```

**Output:**
```yaml
version: 2
models:
  - name: users
    description: "Registry of all customer accounts created via the mobile app."
    columns:
      - name: email
        description: "Primary contact email; must be unique per user."
```