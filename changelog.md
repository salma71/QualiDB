# Changelog

[2025-12-29] - [Docs] - Updated `docs/cli.md` to include all CLI commands and options from `db_check.py`.
[2025-12-29] - [Docs] - Formatted 'Checks' section in `docs/cli.md` for better readability.
[2025-12-29] - [Docs] - Added Mermaid flow diagram to `docs/architecture.md`.
[2025-12-29] - [Feat] - Enabled Mermaid.js support in Jekyll site via `_includes/head_custom.html`.
[2025-12-29] - [Fix] - Upgraded Mermaid initialization to use explicit `mermaid.run()` for robust handling of transformed code blocks.
[2025-12-29] - [Refactor] - Updated Mermaid initialization script to use theme-native loading and custom styling.
[2025-12-29] - [Fix] - Updated Mermaid initialization to correctly parse Jekyll/Rouge code blocks.

[2026-01-14] - [Feat] - Added support for BigQuery views in `db_check.py`. Updated schema generation to include views and their metadata, and ensured `push-to-db` uses `ALTER VIEW` for view descriptions.
[2026-01-14] - [Fix] - Fixed `UnboundLocalError: item_type` in `generate_schema` by correctly scoping variable definitions within the table processing loop.

[2026-01-16] - [Docs] - Provided instructions for using `push-to-db` with custom schema files (e.g., `models/dosedaily_prod_schema.yml`) to sync AI-generated descriptions back to BigQuery.
[2026-01-16] - [Feat] - Added selective table pushing to `push-to-db` via `--table` flag and interactive checkbox menu.
[2026-01-16] - [Feat] - Added `prune-schema` command to remove deleted datasets and tables from the YAML schema file based on database state.
[2026-01-16] - [Feat] - Added `refresh-schema` command for "Reverse Sync," automatically updating the YAML file with new column names and pulling descriptions from the database.

[2026-01-20] - [Fix] - Further optimized `map_subs_id_pyment_ids.sql` (v4) with strict partition pruning, early line item reduction, and pre-normalized CTE joins for maximum performance.
[2026-01-22] - [Feat] - Added `identify-keys` command to `db_check.py`. Implemented heuristic key discovery (suggestions) and BigQuery `INFORMATION_SCHEMA` metadata retrieval. Integrated `polars` for robust reporting and SQL generation.

[2026-02-05] - [Feat] - Added `--schema` and `--append-only` flags to `generate-schema` for non-interactive and cumulative schema documentation.
[2026-02-05] - [Feat] - Created `extract_keys_from_yaml.py` to extract PK/FK metadata from AI-generated descriptions when database metadata access is restricted.
[2026-02-05] - [Fix] - Improved `identify-keys` to support cross-project BigQuery datasets and corrected billing project logic for restricted environments.
[2026-02-05] - [Docs] - Consolidated Skio API findings and technical mappings into `models/Skio_CDP_Final_Implementation.md`.
[2026-02-05] - [Docs] - Finalized `Skio_Final_Discovery_Confirmation_Draft.md` for Google Docs upload, covering schema verification, PK/FK confirmation, and dunning logic validation for the Skio technical team.
[2026-02-06] - [Docs] - Updated `Skio_Final_Discovery_Confirmation_Draft.md` to include `Orders` in the master query for charge verification and added a new section (Section 6) investigating `AuditLog` unreliability for 'Billing attempted' events.
[2026-02-06] - [Docs] - Synchronized the master query in the discovery draft with the full CDP implementation query (adding `nextBillingDate`, `BillingPolicy`, `paymentMethod`, etc.) to ensure all core attributes are verified with the Skio team.
[2026-02-06] - [Docs] - Finalized `Skio_Final_Discovery_Confirmation_Draft.md` with Section 7 on BigQuery cross-join constraints and IAM/data sharing inquiries, and updated gist links for schema verification.
[2026-02-06] - [Docs] - Promoted BigQuery Infrastructure & Query Constraints to Section 2 in `Skio_Final_Discovery_Confirmation_Draft.md` and marked it as **URGENT** to prioritize resolving cross-join limitations.
[2026-02-06] - [Docs] - Added a formal proposal for 4-hour data replication to Section 2 of `Skio_Final_Discovery_Confirmation_Draft.md` to bypass current BigQuery cross-join limitations.
[2026-02-06] - [Docs] - Added Section 5A update to address the "Invisible Webhooks" issue where `subscriptionWillRenew` is active in the dashboard but missing from BQ/API logs.
[2026-02-06] - [Docs] - Updated `models/DE-Klaviyo + Skio_ Event setup for CDP Source of Truth.md` to incorporate critical findings from the Skio Discovery phase, including the "Gross vs. Net Price" discrepancy and gaps in "Agent-Initiated Cancellations."
[2026-02-06] - [Feat] - Created `replicate_skio_tables.py` as a standalone Cloud Function-ready script. Features:
    - Idempotent `MERGE` replication from Skio to `dosedaily-raw`.
    - Auto-partitioning on `createdAt` and clustering logic for large tables (`Subscription`, `Order`, `BillingAttempt`, etc.).
    - Built-in Slack alerting and Metadata sync.
    - Removed `replicate-skio` from `db_check.py` to decouple orchestration logic.

[2026-02-07] - [Docs] - Switched Mermaid formatted diagrams to Live Editor links in `models/Skio_Final_Discovery_Confirmation_Draft.md` for better shareability.
[2026-02-07] - [Feature] - Added Section 9 "Proposed CDP Data Flow" to `models/Skio_Final_Discovery_Confirmation_Draft.md` with a Mermaid diagram illustrating the Skio -> BQ (Replica) -> CDP (Klaviyo/RudderStack) architecture and gap resolution strategy.




[2026-02-07] - [Docs] - Added request for `SkuSwap` object definition details (table vs. API object) to `models/Skio_Final_Discovery_Confirmation_Draft.md` to ensure CDP can track product swaps.
