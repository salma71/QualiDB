import click
import yaml
import polars as pl
import os
import datetime
import re
from sqlalchemy import create_engine, inspect, text
from tabulate import tabulate
import openai
import requests
import anthropic
import questionary
import sys
import warnings 
from sqlalchemy import exc as sa_exc
from dotenv import load_dotenv 
from google.cloud import secretmanager
from questionary import Separator
import json
import uuid

# Try importing uuid6, fallback if missing
try:
    import uuid6
    HAS_UUID6 = True
except ImportError:
    HAS_UUID6 = False

# Try importing BigQuery Client for advanced schema updates
try:
    from google.cloud import bigquery
    HAS_BQ_CLIENT = True
except ImportError:
    HAS_BQ_CLIENT = False

warnings.filterwarnings("ignore", category=sa_exc.SAWarning) 
load_dotenv()

# ==========================================
# 🛠️ DB & SYSTEM HELPERS
# ==========================================

def validate_connection(connection_string):
    if not connection_string or not connection_string.strip():
        raise ValueError("Connection string cannot be empty")
    try:
        engine = create_engine(connection_string, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        click.secho(f"❌ Connection Failed: {e}", fg='red')
        sys.exit(1)

def save_results_to_db(engine, results, dataset_name):
    if not results:
        print("⚠️ No results to save.")
        return

    try:
        run_id = str(uuid6.uuid7()) if HAS_UUID6 else str(uuid.uuid4())
    except Exception:
        run_id = str(uuid.uuid4())
    
    safe_dataset = dataset_name.replace("`", "")
    target_table = f"`{safe_dataset}.data_quality_logs`"

    create_table_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {target_table} (
            run_id STRING,
            run_at TIMESTAMP,
            dataset_name STRING,
            table_name STRING,
            column_name STRING,
            check_type STRING,
            status STRING,
            result_value STRING,
            metadata STRING
        );
    """)
    
    insert_sql = text(f"""
        INSERT INTO {target_table} 
        (run_id, run_at, dataset_name, table_name, column_name, check_type, status, result_value)
        VALUES (:run_id, CURRENT_TIMESTAMP(), :dataset, :table, :col, :check, :status, :val)
    """)
    
    print(f"\n💾 Saving {len(results)} results to {target_table} (Run ID: {run_id})...")
    
    try:
        with engine.begin() as conn:
            conn.execute(create_table_sql)
            for row in results:
                if len(row) < 5: continue
                conn.execute(insert_sql, {
                    "run_id": run_id,
                    "dataset": safe_dataset,
                    "table": row[0],
                    "col": row[1],
                    "check": row[2],
                    "val": row[3],
                    "status": row[4]
                })
        print("✅ Results saved successfully!")
    except Exception as e:
        print(f"❌ Failed to save logs: {e}")

def send_slack_alert(webhook_url, results, dataset_name):
    if not webhook_url or not webhook_url.strip(): return

    fails = [r for r in results if "FAIL" in r[4] or "❌" in r[4]]
    warns = [r for r in results if "OLD" in r[4] or "⚠️" in r[4]]
    passes = [r for r in results if r not in fails and r not in warns]
    
    status_emoji = "🚨" if len(fails) > 0 else ("⚠️" if len(warns) > 0 else "✅")
    
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"{status_emoji} Data Quality Report", "emoji": True}},
        {"type": "divider"}
    ]

    def add_section(items, title, emoji):
        if not items: return
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{emoji} {title}*"}})
        for row in items[:10]: 
            table_name = row[0].split('.')[-1]
            col_name = row[1]
            check_type = row[2]
            val = row[3]
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"• *{table_name}*\n   ↳ {check_type} on `{col_name}`: *{val}*"}
            })

    if fails: add_section(fails, "Critical Failures", "❌")
    if warns: add_section(warns, "Warnings", "⚠️")
    
    if not fails and not warns:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"✅ *All {len(passes)} checks passed successfully.*"}})
    else:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"{len(passes)} checks passed."}]})

    try:
        requests.post(webhook_url, json={"blocks": blocks}, timeout=10)
        print("🔔 Slack notification sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send Slack alert: {e}")

def get_secret(secret_id, project_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    try:
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception: return None

def get_bq_quoted_name(schema, table):
    if not schema: return f"`{table}`"
    parts = schema.split('.')
    quoted_parts = [f"`{p}`" for p in parts]
    quoted_parts.append(f"`{table}`")
    return ".".join(quoted_parts)

def clean_ai_response(text):
    if not text: return ""
    text = re.sub(r'^\s*#+ .*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*.*?\*\*[:\s]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Description:\s*', '', text, flags=re.MULTILINE)
    return " ".join(text.split()).strip()

# ==========================================
# 🔧 BIGQUERY NESTED UPDATE LOGIC
# ==========================================

def update_bq_nested_schema(project_id, dataset, table, column_path, description):
    """
    Recursively updates the schema of a BigQuery table using the Python Client.
    Required because SQL 'ALTER COLUMN' doesn't support nested fields.
    """
    if not HAS_BQ_CLIENT:
        return f"Skipping {column_path}: 'google-cloud-bigquery' not installed."

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table}"
    
    try:
        bq_table = client.get_table(table_ref)
    except Exception as e:
        return f"Could not fetch table {table_ref}: {str(e).splitlines()[0]}"

    path_parts = column_path.split('.')
    
    def update_field_list(schema_fields, parts):
        new_schema = []
        found = False
        target = parts[0]
        
        for field in schema_fields:
            if field.name == target:
                found = True
                if len(parts) == 1:
                    # Found the target field, update description
                    new_field = field.to_api_repr()
                    new_field['description'] = description
                    new_schema.append(bigquery.SchemaField.from_api_repr(new_field))
                else:
                    # Need to recurse deeper
                    new_sub_fields = update_field_list(field.fields, parts[1:])
                    new_field = field.to_api_repr()
                    new_field['fields'] = [f.to_api_repr() for f in new_sub_fields]
                    new_schema.append(bigquery.SchemaField.from_api_repr(new_field))
            else:
                new_schema.append(field)
        
        return new_schema

    # Update Schema in Memory
    new_schema = update_field_list(bq_table.schema, path_parts)
    bq_table.schema = new_schema
    
    # Push Update
    try:
        client.update_table(bq_table, ["schema"])
        return None
    except Exception as e:
        return f"BQ Schema Update Failed ({column_path}): {str(e).splitlines()[0]}"

def update_db_description(connection, schema, table, column, description, dialect, engine=None, table_type='table'):
    if not description or "Error" in description: return None
    
    safe_desc = description.replace("'", "''")
    
    try:
        # --- BIGQUERY SPECIAL HANDLING ---
        if dialect == 'bigquery':
            # Check if it's a nested column (contains dot)
            if column and "." in column:
                # Use Python Client API for nested fields
                project_id = engine.url.host or os.getenv("GOOGLE_CLOUD_PROJECT")
                if not project_id:
                    return f"Skipping {column}: Cannot determine Google Project ID."
                
                # BigQuery schemas usually don't have dots in dataset names in the API
                # Clean the schema if it came in as "project.dataset"
                clean_dataset = schema.split('.')[-1]
                
                return update_bq_nested_schema(project_id, clean_dataset, table, column, description)

            # Standard Top-Level Column Update (Use SQL)
            safe_desc_bq = description.replace('"', '\\"')
            target = get_bq_quoted_name(schema, table)
            
            # Use ALTER VIEW for views and materialized views in BigQuery
            cmd = "ALTER TABLE"
            if table_type in ['view', 'materialized_view']:
                cmd = "ALTER VIEW"
                
            if column:
                sql = f'{cmd} {target} ALTER COLUMN `{column}` SET OPTIONS(description="{safe_desc_bq}")'
            else:
                sql = f'{cmd} {target} SET OPTIONS(description="{safe_desc_bq}")'
        
        # --- POSTGRES / SNOWFLAKE ---
        elif dialect in ['postgresql', 'snowflake']:
            target = f"{schema}.{table}.{column}" if column else f"{schema}.{table}"
            obj_type = "COLUMN" if column else "TABLE"
            sql = f"COMMENT ON {obj_type} {target} IS '{safe_desc}'"
        else: return None
        
        connection.execute(text(sql))
        return None
        
    except Exception as e:
        # Clean up error message for display
        err_msg = str(e).split('\n')[0]
        # click.secho(f"⚠️ Failed to write to DB ({table}.{column}): {err_msg}", fg='yellow')
        return err_msg

def get_inspector(connection_string):
    try:
        engine = validate_connection(connection_string)
        return inspect(engine), engine
    except Exception as e:
        click.secho(f"Error connecting: {e}", fg='red'); sys.exit(1)

# ==========================================
# 🧠 AI ENGINE (Unchanged)
# ==========================================

def get_valid_google_model(api_key):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            for m in response.json().get('models', []):
                if "gemini" in m['name'] and "generateContent" in m.get('supportedGenerationMethods', []):
                    return m['name'].replace("models/", "")
    except: pass
    return "gemini-pro"

def get_valid_claude_model(client, preferred_keyword="sonnet"):
    try:
        models = client.models.list()
        available_ids = [m.id for m in models.data]
        for m_id in available_ids:
            if preferred_keyword in m_id and "3-5" in m_id: return m_id
        for m_id in available_ids:
            if preferred_keyword in m_id: return m_id
        if available_ids: return available_ids[0]
    except Exception: pass
    return "claude-3-sonnet-20240229"

def generate_ai_description(item_name, item_type, model_name, context=""):
    prompt_text = (
        f"Write a single, concise 1-sentence business description for the {item_type} '{item_name}'. "
        f"Context: {context}. "
        f"Rules: Return ONLY the description text. Do NOT use Markdown, headers, or labels like 'Description:'."
    )
    model_lower = model_name.lower()

    if "claude" in model_lower or "sonnet" in model_lower:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key: return "ERROR: Missing ANTHROPIC_API_KEY"
        try:
            client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
            print(f"  > 🤖 Asking Claude ({model_name}) about {item_name}...", end="\r")
            candidates = ["claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022"] if "sonnet" in model_lower else [model_name]
            last_error = ""
            for target in candidates:
                try:
                    msg = client.messages.create(model=target, max_tokens=100, messages=[{"role": "user", "content": prompt_text}])
                    print(" " * 80, end="\r")
                    return clean_ai_response(msg.content[0].text)
                except anthropic.NotFoundError:
                    last_error = f"Model {target} not found"
                    continue 
                except Exception as e:
                    last_error = repr(e) 
                    break 
            print(" " * 80, end="\r")
            return f"Claude Error: {last_error}"
        except Exception as e: return f"Claude Init Error: {repr(e)}"

    elif "gemini" in model_lower:
        print(f"  > 🤖 Asking Gemini about {item_name}...", end="\r")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: return "ERROR: Missing GOOGLE_API_KEY"
        target = "gemini-1.5-flash" if model_name in ["gemini", "gemini-pro"] else model_name
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        try:
            resp = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt_text}]}]}, timeout=15)
            if resp.status_code == 404:
                new_model = get_valid_google_model(api_key)
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{new_model}:generateContent?key={api_key}"
                resp = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt_text}]}]}, timeout=15)
            print(" " * 80, end="\r")
            if resp.status_code == 200: return clean_ai_response(resp.json()['candidates'][0]['content']['parts'][0]['text'])
            return f"Error {resp.status_code}"
        except Exception as e: return f"Conn Error: {str(e)}"

    else:
        print(f"  > 🤖 Asking OpenAI about {item_name}...", end="\r")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: return "ERROR: Missing OPENAI_API_KEY"
        try:
            client = openai.OpenAI(api_key=api_key, timeout=20.0)
            resp = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt_text}], max_tokens=40)
            print(" " * 80, end="\r")
            return clean_ai_response(resp.choices[0].message.content)
        except Exception as e: return f"OpenAI Error: {str(e)}"

# ==========================================
# 🛠️ CHECK LOGIC (Unchanged)
# ==========================================

def check_nulls(connection, schema, table, column, **kwargs):
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column): return ("Skipped (Bad Name)", "⚠️")
    try:
        tbl_ref = get_bq_quoted_name(schema, table)
        query = text(f"SELECT COUNT(*) FROM {tbl_ref} WHERE {column} IS NULL")
        count = connection.execute(query).scalar()
        return ("PASS", "✅") if count == 0 else (f"FAIL ({count})", "Failed")
    except: return None

def check_uniqueness(connection, schema, table, column, **kwargs):
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column): return ("Skipped (Bad Name)", "⚠️")
    try:
        tbl_ref = get_bq_quoted_name(schema, table)
        query = text(f"SELECT COUNT({column}) - COUNT(DISTINCT {column}) FROM {tbl_ref}")
        diff = connection.execute(query).scalar()
        return ("PASS", "✅") if diff == 0 else (f"{diff} duplicates", "Failed")
    except: return None

def check_freshness(connection, schema, table, column, **kwargs):
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column): return ("Skipped (Bad Name)", "⚠️")
    col_lower = column.lower()
    dtype = str(kwargs.get('dtype', '')).lower()
    time_terms = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'ingested', '_at', '_ts']
    if not any(t in dtype for t in ['date', 'time', 'timestamp']) and not any(x in col_lower for x in time_terms): return None 
    if 'status' in col_lower: return None

    try:
        tbl_ref = get_bq_quoted_name(schema, table)
        query = text(f"SELECT MAX({column}) FROM {tbl_ref}")
        last_update = connection.execute(query).scalar()
        if not last_update: return ("EMPTY", "⚪")
        
        if isinstance(last_update, str):
            try: 
                clean_ts = str(last_update).replace('Z', '+00:00')
                last_update = datetime.datetime.fromisoformat(clean_ts)
            except: return (f"Bad fmt: {str(last_update)[:10]}...", "⚠️")

        if isinstance(last_update, datetime.date) and not isinstance(last_update, datetime.datetime):
             last_update = datetime.datetime.combine(last_update, datetime.datetime.min.time())

        if last_update.tzinfo:
            now = datetime.datetime.now(last_update.tzinfo)
        else:
            now = datetime.datetime.now()

        diff = now - last_update
        hours = diff.total_seconds() / 3600
        if hours < 0: return (f"Future: {abs(hours):.1f}h", "⚠️")
        return (f"{hours:.1f}h ago", "✅") if hours < 24 else (f"{hours:.1f}h ago", "⚠️ OLD")
    except Exception as e:
        error_str = str(e)
        if "404" in error_str: return ("API 404 (Check SQL)", "❌")
        return (f"Err: {error_str[:30]}", "❌")

def get_ai_suggested_config(model_name, table_name, columns):
    prompt = (
        f"I have a table '{table_name}' with columns: {columns}. "
        "Return a JSON object with keys 'freshness_col' (best timestamp) "
        "and 'completeness_col' (best ID/PK). Return null if none found. "
        "Return ONLY JSON. No markdown."
    )
    try:
        response_text = generate_ai_description(table_name, "config", model_name, context=prompt)
        if not response_text: return None
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception: return None 

# ==========================================
# 🖥️ CLI SUB-COMMANDS (Unchanged)
# ==========================================
@click.command(name='generate-schema')
@click.option('--conn', prompt='Connection String')
@click.option('--output', default='models/schema.yml')
@click.option('--model', default='gpt-3.5-turbo')
@click.option('--write-db', is_flag=True, help="Write descriptions back to DB")
@click.option('--schema', help="Specific schema(s) to scan (comma-separated)")
@click.option('--append-only', is_flag=True, help="Only add new tables/columns, do not overwrite existing descriptions")
def generate_schema(conn, output, model, write_db, schema, append_only):
    inspector, engine = get_inspector(conn)
    
    # ==========================================
    # 1. LOAD EXISTING YAML (Load once)
    # ==========================================
    schema_data = {"version": 2, "sources": []}
    existing_desc_map = {}
    
    if os.path.exists(output):
        try:
            with open(output, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml: schema_data = loaded_yaml
                # Map existing descriptions to avoid re-generating known ones
                for src in schema_data.get('sources', []):
                    for t in src.get('tables', []):
                        t_name = t['name']
                        if 'description' in t: existing_desc_map[t_name] = t['description']
                        for c in t.get('columns', []):
                            existing_desc_map[f"{t_name}.{c['name']}"] = c.get('description')
        except Exception as e:
            print(f"⚠️ Warning loading existing YAML: {e}")

    # ==========================================
    # 2. SELECT SCHEMAS (Multi-select)
    # ==========================================
    available_schemas = inspector.get_schema_names()
    
    # Options logic
    if schema:
        selected_choices = [s.strip() for s in schema.split(',')]
    else:
        choices = ["(Select All)", "📍 Manually enter"] + sorted(available_schemas)
        
        selected_choices = questionary.checkbox(
            "Select Schemas/Datasets to document:",
            choices=choices
        ).ask()

    if not selected_choices:
        print("No schemas selected. Exiting.")
        return

    target_schemas_list = []

    # Handle special choices
    if "(Select All)" in selected_choices:
        target_schemas_list = available_schemas
    else:
        # Filter out the UI elements to get actual schema names
        target_schemas_list = [s for s in selected_choices if s not in ["(Select All)", "📍 Manually enter"]]
        
        # Handle Manual Entry
        if "📍 Manually enter" in selected_choices:
            manual_input = click.prompt("Enter dataset IDs (comma separated, e.g. project.dataset)")
            manual_schemas = [s.strip() for s in manual_input.split(',') if s.strip()]
            target_schemas_list.extend(manual_schemas)

    # Remove duplicates just in case
    target_schemas_list = list(set(target_schemas_list))
    
    print(f"\n🚀 Starting documentation for {len(target_schemas_list)} schemas...\n")

    # ==========================================
    # 3. LOOP THROUGH SCHEMAS
    # ==========================================
    for target_schema in target_schemas_list:
        click.secho(f"📂 Processing Schema: {target_schema}", bold=True, fg='cyan')

        # --- Get Tables and Views for this Schema ---
        all_items = [] # List of (name, type)
        
        # Check if this is a cross-project dataset (has a dot, like 'project.dataset')
        if "." in target_schema and HAS_BQ_CLIENT:
            try:
                billing_project = engine.url.host 
                client = bigquery.Client(project=billing_project)
                bq_tables = client.list_tables(target_schema)
                for t in bq_tables:
                    t_type = t.table_type.lower() if hasattr(t, 'table_type') else 'table'
                    all_items.append((t.table_id, t_type))
            except Exception as e:
                print(f"   ⚠️ Native listing failed: {e}. Falling back to Inspector.")
                try: 
                    all_items = [(t, 'table') for t in inspector.get_table_names(schema=target_schema)]
                    all_items.extend([(v, 'view') for v in inspector.get_view_names(schema=target_schema)])
                except: pass
        else:
            # Standard local listing
            try: 
                all_items = [(t, 'table') for t in inspector.get_table_names(schema=target_schema)]
                all_items.extend([(v, 'view') for v in inspector.get_view_names(schema=target_schema)])
            except: pass

        if not all_items:
            print(f"   ❌ No tables or views found in '{target_schema}'. Skipping.")
            continue

        # Prepare choices for UI
        choices_map = {}
        for name, t_type in all_items:
            label = name
            if t_type == 'view': label = f"{name} (view)"
            elif t_type == 'materialized_view': label = f"{name} (m-view)"
            choices_map[label] = (name, t_type)

        # --- Select Tables for this specific Schema ---
        if schema:
            selected_labels = ["(Select All)"]
        else:
            selected_labels = questionary.checkbox(
                f"Select tables/views in '{target_schema}':", 
                choices=["(Select All)"] + sorted(choices_map.keys())
            ).ask()
        
        if not selected_labels:
            print("   Attributes skipped.")
            continue

        final_items = all_items if "(Select All)" in selected_labels else [choices_map[label] for label in selected_labels]
        
        # Mapping for easier lookup during processing
        final_item_types = {name: t_type for name, t_type in final_items}
        final_tables = [name for name, _ in final_items]

        # --- Locate/Create Source in Data Structure ---
        target_source = None
        for src in schema_data.get("sources", []):
            if src.get("name") == target_schema:
                target_source = src
                break
        
        if not target_source:
            target_source = {"name": target_schema, "tables": []}
            schema_data.setdefault("sources", []).append(target_source)

        current_tables_dict = {t['name']: t for t in target_source.get("tables", [])}
        overwrite_all = False if append_only else None 

        # --- Process Tables ---
        with click.progressbar(final_tables, label=f'   Generating {target_schema}') as bar:
            for table in bar:
                clean_table_name = table.split('.')[-1]
                
                # Construct query-able name
                if "." in target_schema: full_table_ref = f"{target_schema}.{clean_table_name}"
                else: full_table_ref = table

                # 1. TABLE DESCRIPTION
                t_desc = existing_desc_map.get(table)
                item_type = final_item_types.get(table, 'table')
                should_gen_table = True
                
                if t_desc:
                    if overwrite_all is True: should_gen_table = True
                    elif overwrite_all is False: should_gen_table = False
                    else:
                        # Pause progress bar to ask question
                        print(f"\n   Example: Table '{table}' has desc: '{t_desc[:30]}...'")
                        choice = questionary.select("   Overwrite?", choices=["Yes", "No", "Yes to All", "No to All"]).ask()
                        if choice == "Yes": should_gen_table = True
                        elif choice == "No": should_gen_table = False
                        elif choice == "Yes to All": should_gen_table = True; overwrite_all = True
                        elif choice == "No to All": should_gen_table = False; overwrite_all = False

                final_t_desc = t_desc
                if should_gen_table:
                    final_t_desc = generate_ai_description(clean_table_name, item_type, model)
                    if write_db:
                        try:
                            with engine.begin() as c: 
                                update_db_description(c, target_schema, clean_table_name, None, final_t_desc, engine.dialect.name, engine=engine, table_type=item_type)
                        except Exception as e:
                            # Use print instead of click.echo to not break progress bar layout too much
                            pass 

                # 2. COLUMN DESCRIPTIONS
                cols_data = []
                try: 
                    columns = inspector.get_columns(clean_table_name, schema=target_schema)
                except: 
                    try: columns = inspector.get_columns(full_table_ref)
                    except: columns = []

                for col in columns:
                    c_name = col['name']
                    c_key = f"{table}.{c_name}"
                    c_desc = existing_desc_map.get(c_key)
                    should_gen_col = True
                    
                    if c_desc and overwrite_all is False: should_gen_col = False
                    elif c_desc and overwrite_all is True: should_gen_col = True
                    elif c_desc: should_gen_col = False 
                    
                    final_c_desc = c_desc
                    if should_gen_col:
                        final_c_desc = generate_ai_description(c_name, "column", model, f"Table: {clean_table_name}")
                        if write_db:
                            try:
                                with engine.begin() as c: 
                                    update_db_description(c, target_schema, clean_table_name, c_name, final_c_desc, engine.dialect.name, engine=engine, table_type=item_type)
                            except: pass
                    
                    cols_data.append({
                        "name": c_name, 
                        "description": final_c_desc, 
                        "data_type": str(col['type'])
                    })
                
                # Update the source dictionary
                current_tables_dict[table] = {
                    "name": table, 
                    "type": item_type,
                    "description": final_t_desc, 
                    "columns": cols_data
                }

        # Update the master list for this specific source
        target_source["tables"] = list(current_tables_dict.values())
        
        # Save after every schema to prevent data loss on long runs
        with open(output, 'w') as f: yaml.dump(schema_data, f, sort_keys=False)

    click.secho(f"\n✅ All tasks finished. Merged into {output}", fg='green')
@click.command(name='check-quality')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
def check_quality(conn):
    if not conn: conn = click.prompt("Connection String")
    try: engine = validate_connection(conn)
    except Exception as e: print(f"Connection Error: {e}"); return
    
    config_path = "checks.yml"
    if not os.path.exists(config_path): print("❌ No 'checks.yml' found. Please run 'generate-config' first."); return
    with open(config_path, 'r') as f: config = yaml.safe_load(f) or {}
    checks_to_run = config.get('checks', [])
    if not checks_to_run: print("⚠️ Config file is empty."); return

    print(f"Running {len(checks_to_run)} configured checks... 🏃")
    results = []
    default_schema = checks_to_run[0]['table'].split('.')[0] if '.' in checks_to_run[0]['table'] else "public"

    with engine.connect() as connection:
        with click.progressbar(checks_to_run, label="Scanning") as bar:
            for check in bar:
                full_table_name = check['table']
                schema_part, table_part = full_table_name.split('.', 1) if '.' in full_table_name else (default_schema, full_table_name)

                if 'freshness_col' in check:
                    col = check['freshness_col']
                    res = check_freshness(connection, schema_part, table_part, col)
                    if res: results.append([full_table_name, col, "Freshness", res[0], res[1]])
                    else: results.append([full_table_name, col, "Freshness", "SQL Error", "Failed"])

                if 'completeness_col' in check:
                    col = check['completeness_col']
                    res = check_nulls(connection, schema_part, table_part, col)
                    if res: results.append([full_table_name, col, "Completeness", res[0], res[1]])
                    else: results.append([full_table_name, col, "Completeness", "SQL Error", "Failed"])

    print("\n" + tabulate(results, headers=["Table", "Column", "Check", "Result", "Status"], tablefmt="simple_grid"))
    if click.confirm("\n💾 Save results to DB?"): save_results_to_db(engine, results, default_schema)
    if click.confirm("🔔 Send report to Slack?"):
        webhook_url = get_secret("SLACK_WEBHOOK_URL", "dosedaily-raw")
        if webhook_url: send_slack_alert(webhook_url, results, default_schema)
        else: print("❌ Could not fetch Slack Webhook.")

@click.command(name='push-to-db')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
@click.option('--input', default='models/schema.yml', help='Path to schema.yml')
@click.option('--table', 'target_tables', multiple=True, help='Specific table(s) to push (schema.table)')
def push_to_db(conn, input, target_tables):
    if not conn: conn = click.prompt("Connection String")
    if not os.path.exists(input): click.secho(f"❌ File not found: {input}", fg='red'); return

    inspector, engine = get_inspector(conn)
    dialect = engine.dialect.name
    with open(input, 'r') as f: data = yaml.safe_load(f) or {}
    sources = data.get('sources', [])
    if not sources: click.echo("No sources found."); return

    # --- Selective Filtering ---
    if target_tables:
        filtered_sources = []
        for src in sources:
            src_name = src.get('name')
            new_tables = [t for t in src.get('tables', []) if t['name'] in target_tables or f"{src_name}.{t['name']}" in target_tables]
            if new_tables:
                src_copy = src.copy()
                src_copy['tables'] = new_tables
                filtered_sources.append(src_copy)
        sources = filtered_sources
        if not sources:
            click.secho(f"❌ No matching tables found for selection: {target_tables}", fg='red')
            return

    total_items = 0
    for src in sources:
        for tbl in src.get('tables', []): total_items += 1 + len(tbl.get('columns', []))
    click.echo(f"Found {total_items} items to push. Pushing to {dialect.upper()}... 🚀")

    broken_views = set()
    errors = []

    with engine.connect() as connection:
        with connection.begin(): 
            with click.progressbar(length=total_items, label='Syncing') as bar:
                for src in sources:
                    yaml_schema = src.get('name') or src.get('schema') or 'public'
                    db_target_schema = yaml_schema
                    if 'bigquery' in dialect:
                         if '.' not in yaml_schema and engine.url.host: db_target_schema = f"{engine.url.host}.{yaml_schema}"

                    for tbl in src.get('tables', []):
                        raw_table_name = tbl['name']
                        final_table_name = raw_table_name.split('.')[-1] if '.' in raw_table_name else raw_table_name
                        table_desc = tbl.get('description')
                        table_type = tbl.get('type', 'table')
                        
                        full_table_path = f"{db_target_schema}.{final_table_name}"

                        if table_desc:
                            err = update_db_description(connection, db_target_schema, final_table_name, None, table_desc, dialect, engine=engine, table_type=table_type)
                            if err:
                                if "400" in err or "Unrecognized name" in err or "failed to parse" in err:
                                    broken_views.add(full_table_path)
                                else:
                                    errors.append(f"❌ {full_table_path}: {err}")

                        bar.update(1)
                        for col in tbl.get('columns', []):
                            col_name = col['name']
                            col_desc = col.get('description')
                            if col_desc:
                                err = update_db_description(connection, db_target_schema, final_table_name, col_name, col_desc, dialect, engine=engine, table_type=table_type)
                                if err:
                                    if "400" in err or "Unrecognized name" in err or "failed to parse" in err:
                                        broken_views.add(full_table_path)
                                    else:
                                        errors.append(f"❌ {full_table_path}.{col_name}: {err}")
                            bar.update(1)

    if broken_views or errors:
        click.echo("\n" + "="*50)
        if broken_views:
            click.secho("🚩 BROKEN VIEWS DETECTED", fg='red', bold=True)
            click.echo("The following views have invalid SQL and could not be updated:")
            for v in sorted(broken_views):
                click.secho(f"  • {v}", fg='red')
            click.echo("\n💡 Tip: Fix these views in BigQuery to enable metadata sync.")

        if errors:
            click.secho("\n⚠️ OTHER ERRORS", fg='yellow', bold=True)
            for e in errors[:10]:
                click.echo(f"  {e}")
            if len(errors) > 10:
                click.echo(f"  ... and {len(errors)-10} more.")
        click.echo("="*50)
    else:
        click.secho(f"\n✅ Successfully pushed all descriptions to database!", fg='green')

@click.command(name='prune-schema')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
@click.option('--input', default='models/schema.yml', help='Path to schema.yml')
@click.option('--output', help='Path to save pruned YAML (defaults to input)')
def prune_schema(conn, input, output):
    if not conn: conn = click.prompt("Connection String")
    if not os.path.exists(input): click.secho(f"❌ File not found: {input}", fg='red'); return
    if not output: output = input

    inspector, engine = get_inspector(conn)
    dialect = engine.dialect.name
    with open(input, 'r') as f: data = yaml.safe_load(f) or {}
    sources = data.get('sources', [])
    if not sources: click.echo("No sources found."); return

    try:
        db_schemas = inspector.get_schema_names()
    except Exception as e:
        click.secho(f"❌ Could not fetch schemas from DB: {e}", fg='red')
        return

    pruned_sources = []
    click.echo(f"Found {len(sources)} sources in YAML. Validating against DB... 🔍")
    
    for src in sources:
        src_name = src.get('name') or src.get('schema')
        if not src_name: continue
        
        # BigQuery handling
        clean_src = src_name.split('.')[-1] if '.' in src_name else src_name
        
        exists = False
        if src_name in db_schemas or clean_src in db_schemas:
            exists = True
        elif 'bigquery' in dialect and '.' not in src_name and engine.url.host:
             if f"{engine.url.host}.{src_name}" in db_schemas: exists = True

        if not exists:
            click.secho(f"  🗑️ Removing missing dataset: {src_name}", fg='yellow')
            continue
            
        # Check tables
        try:
            db_tables = inspector.get_table_names(schema=src_name)
            db_views = inspector.get_view_names(schema=src_name)
            all_db_items = set(db_tables + db_views)
        except:
            click.secho(f"  ⚠️ Could not fetch tables for {src_name}. Skipping.", fg='yellow')
            pruned_sources.append(src)
            continue
            
        current_tables = src.get('tables', [])
        valid_tables = []
        for tbl in current_tables:
            tbl_name = tbl['name']
            clean_tbl_name = tbl_name.split('.')[-1] if '.' in tbl_name else tbl_name
            
            if clean_tbl_name in all_db_items or tbl_name in all_db_items:
                valid_tables.append(tbl)
            else:
                click.secho(f"  🗑️ Removing missing table: {src_name}.{tbl_name}", fg='yellow')
        
        if valid_tables:
            src['tables'] = valid_tables
            pruned_sources.append(src)
        else:
            click.secho(f"  🗑️ Removing empty dataset: {src_name}", fg='yellow')

    if pruned_sources:
        data['sources'] = pruned_sources
        with open(output, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        click.secho(f"✅ Pruned YAML saved to {output}", fg='green')
    else:
        click.secho("⚠️ No sources left after pruning!", fg='red')

@click.command(name='refresh-schema')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
@click.option('--input', default='models/schema.yml', help='Path to schema.yml')
@click.option('--output', help='Path to save refreshed YAML (defaults to input)')
def refresh_schema(conn, input, output):
    if not conn: conn = click.prompt("Connection String")
    if not os.path.exists(input): click.secho(f"❌ File not found: {input}", fg='red'); return
    if not output: output = input

    inspector, engine = get_inspector(conn)
    dialect = engine.dialect.name
    with open(input, 'r') as f: data = yaml.safe_load(f) or {}
    sources = data.get('sources', [])
    if not sources: click.echo("No sources found."); return

    click.echo(f"Refreshing schema metadata from {dialect.upper()}... 🔄")
    
    total_changes = 0
    new_cols_total = 0
    desc_updates_total = 0

    for src in sources:
        src_name = src.get('name') or src.get('schema')
        if not src_name: continue
        
        click.echo(f"📂 Scanning: {src_name}")
        
        tables = src.get('tables', [])
        for tbl in tables:
            tbl_name = tbl['name']
            clean_tbl_name = tbl_name.split('.')[-1] if '.' in tbl_name else tbl_name
            
            # Fetch current DB columns
            try:
                db_cols = inspector.get_columns(clean_tbl_name, schema=src_name)
            except:
                try:
                    db_cols = inspector.get_columns(tbl_name, schema=src_name)
                except:
                    click.secho(f"  ⚠️ Could not fetch columns for {tbl_name}", fg='yellow')
                    continue
            
            db_col_map = {c['name']: c for c in db_cols}
            yaml_cols = tbl.get('columns', [])
            yaml_col_names = {c['name'] for c in yaml_cols}
            
            # 1. Update Existing Columns / Pull Descriptions
            for col in yaml_cols:
                c_name = col['name']
                if c_name in db_col_map:
                    db_c = db_col_map[c_name]
                    db_desc = db_c.get('comment') or ""
                    yaml_desc = col.get('description') or ""
                    
                    # If DB has a description and YAML doesn't, or they differ
                    if db_desc and db_desc != yaml_desc:
                        col['description'] = db_desc
                        desc_updates_total += 1
                        total_changes += 1
            
            # 2. Add Missing Columns
            for c_name, db_c in db_col_map.items():
                if c_name not in yaml_col_names:
                    new_col = {
                        "name": c_name,
                        "description": db_c.get('comment') or "",
                        "data_type": str(db_c.get('type', ''))
                    }
                    yaml_cols.append(new_col)
                    new_cols_total += 1
                    total_changes += 1
            
            tbl['columns'] = yaml_cols

    if total_changes > 0:
        with open(output, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        click.secho(f"✅ Refresh Complete!", fg='green', bold=True)
        click.echo(f"   • {new_cols_total} New columns added.")
        click.echo(f"   • {desc_updates_total} Descriptions pulled from DB.")
        click.echo(f"💾 Saved to: {output}")
    else:
        click.secho("✅ YAML is already in sync with DB. No changes needed.", fg='green')

# ==========================================
# 🔑 KEY IDENTIFICATION (PK/FK)
# ==========================================

def get_bq_keys(engine, schema, location=None):
    """Fetch PK/FK metadata from BigQuery INFORMATION_SCHEMA."""
    if engine.dialect.name != 'bigquery':
        return pl.DataFrame()

    # 1. Parse Project and Dataset
    parts = schema.split('.')
    if len(parts) > 1:
        project = parts[0]
        dataset_name = parts[-1]
    else:
        project = engine.url.host or os.getenv("GOOGLE_CLOUD_PROJECT")
        dataset_name = schema

    # 2. Detect Location (Region)
    # Priority: 1. Manual --location flag, 2. Auto-detect from Dataset, 3. None (let BQ decide)
    final_location = location
    if not final_location and HAS_BQ_CLIENT:
        try:
            client = bigquery.Client(project=project)
            ds_ref = client.get_dataset(f"{project}.{dataset_name}")
            final_location = ds_ref.location
        except Exception: 
            final_location = None # Let BigQuery auto-detect during query

    # 3. Construct Queries
    pk_query = f"""
    SELECT 
        k.table_name, 
        k.column_name, 
        'PRIMARY KEY' as constraint_type,
        CAST(NULL AS STRING) as ref_table,
        CAST(NULL AS STRING) as ref_column
    FROM 
        `{project}.{dataset_name}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` k
    JOIN 
        `{project}.{dataset_name}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` c
    ON 
        k.constraint_name = c.constraint_name
        AND k.table_name = c.table_name
    WHERE c.constraint_type = 'PRIMARY KEY'
    """
    
    fk_query = f"""
    SELECT 
        k1.table_name,
        k1.column_name,
        'FOREIGN KEY' as constraint_type,
        k2.table_name as ref_table,
        k2.column_name as ref_column
    FROM 
        `{project}.{dataset_name}.INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS` rc
    JOIN 
        `{project}.{dataset_name}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` k1 
        ON rc.constraint_name = k1.constraint_name
    JOIN 
        `{project}.{dataset_name}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` k2 
        ON rc.unique_constraint_name = k2.constraint_name
    """
    
    try:
        pk_rows, fk_rows = [], []
        if HAS_BQ_CLIENT:
            billing_project = engine.url.host or os.getenv("GOOGLE_CLOUD_PROJECT")
            client = bigquery.Client(project=billing_project)
            # Only pass location if we actually found one/have one
            job_kwargs = {"location": final_location} if final_location else {}
            pk_rows = [dict(r) for r in client.query(pk_query, **job_kwargs).result()]
            fk_rows = [dict(r) for r in client.query(fk_query, **job_kwargs).result()]
        else:
            with engine.connect() as conn:
                pk_rows = [dict(r._mapping) for r in conn.execute(text(pk_query)).fetchall()]
                fk_rows = [dict(r._mapping) for r in conn.execute(text(fk_query)).fetchall()]
        
        data = pk_rows + fk_rows
        return pl.DataFrame(data) if data else pl.DataFrame(schema={
            "table_name": pl.Utf8, "column_name": pl.Utf8, "constraint_type": pl.Utf8, 
            "ref_table": pl.Utf8, "ref_column": pl.Utf8
        })
    except Exception as e:
        err_msg = str(e).splitlines()[0]
        # Ignore "Not found" for INFORMATION_SCHEMA (it mean no constraints at all)
        if "Not found" in err_msg and "INFORMATION_SCHEMA" in err_msg:
             return pl.DataFrame(schema={"table_name": pl.Utf8, "column_name": pl.Utf8, "constraint_type": pl.Utf8, "ref_table": pl.Utf8, "ref_column": pl.Utf8})
        
        if "location" in err_msg.lower():
            click.secho(f"  ⚠️ Location Error for '{dataset_name}': BQ could not find it. Try passing --location explicitly.", fg='red')
        else:
            click.secho(f"  ⚠️ BQ Meta Fetch Failed ({dataset_name}): {err_msg}", fg='yellow')
        return pl.DataFrame()

def suggest_keys(inspector, schema, tables, model=None):
    """Heuristic logic to suggest potential PK/FKs."""
    suggestions = []
    all_table_names = {t.lower(): t for t in tables}
    
    # Common PK patterns
    pk_patterns = ['id', 'uuid', 'uid', 'pk', 'key', 'code', 'checksum', 'hash']
    
    for table in tables:
        try:
            cols = inspector.get_columns(table, schema=schema)
        except: continue
        
        t_lower = table.lower()
        found_pk = False
            
        for col in cols:
            c_name = col['name'].lower()
            
            # 1. Smarter PK Suggestion
            is_pk_candidate = False
            if c_name in pk_patterns: is_pk_candidate = True
            elif c_name == f"{t_lower}_id": is_pk_candidate = True
            elif c_name.endswith('_id') and not found_pk:
                # If it's the first _id column, it's a strong candidate
                is_pk_candidate = True
            elif 'key' in c_name or 'code' in c_name:
                # If the table name is in the column name (e.g. order_item_code)
                if any(part in c_name for part in t_lower.split('_') if len(part) > 3):
                    is_pk_candidate = True

            if is_pk_candidate and not found_pk:
                suggestions.append({
                    "table_name": table,
                    "column_name": col['name'],
                    "constraint_type": "SUGGESTED PK",
                    "ref_table": None,
                    "ref_column": None
                })
                found_pk = True # Suggest one PK per table
            
            # 2. FK Suggestion
            elif c_name.endswith(('_id', '_key', '_code')):
                potential_ref = re.sub(r'(_id|_key|_code)$', '', c_name)
                # Match against other tables
                if potential_ref in all_table_names:
                    suggestions.append({
                        "table_name": table,
                        "column_name": col['name'],
                        "constraint_type": "SUGGESTED FK",
                        "ref_table": all_table_names[potential_ref],
                        "ref_column": "id" # Assume standard id as target
                    })
                else:
                    # Partial match
                    for t_comp_lower, t_orig in all_table_names.items():
                        if t_comp_lower != t_lower:
                            if t_comp_lower.startswith(potential_ref) or potential_ref.startswith(t_comp_lower):
                                suggestions.append({
                                    "table_name": table,
                                    "column_name": col['name'],
                                    "constraint_type": "SUGGESTED FK",
                                    "ref_table": t_orig,
                                    "ref_column": "id"
                                })
                                break
                                
    return pl.DataFrame(suggestions) if suggestions else pl.DataFrame(schema={
        "table_name": pl.Utf8, "column_name": pl.Utf8, "constraint_type": pl.Utf8, 
        "ref_table": pl.Utf8, "ref_column": pl.Utf8
    })

@click.command(name='identify-keys')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
@click.option('--schema', help='Specific schema to scan')
@click.option('--location', help='BigQuery region (e.g. US, EU, us-east4)')
def identify_keys(conn, schema, location):
    """Identify Primary and Foreign Keys (Existing + Suggested)."""
    if not conn: conn = click.prompt("Connection String")
    inspector, engine = get_inspector(conn)
    
    available_schemas = inspector.get_schema_names()
    target_schemas = [schema] if schema else questionary.checkbox(
        "Select Schemas to identify keys in:",
        choices=sorted(available_schemas)
    ).ask()
    
    if not target_schemas: return

    all_dfs = []
    
    for s in target_schemas:
        click.secho(f"\n🔍 Scanning Schema: {s}", bold=True, fg='cyan')
        
        # 0. Parsing Schema for Inspector (Handle project.dataset)
        clean_dataset = s.split('.')[-1] if '.' in s else s
        
        # 1. System Keys
        sys_keys = pl.DataFrame()
        if engine.dialect.name == 'bigquery':
            sys_keys = get_bq_keys(engine, s, location=location)
            
        # 2. Heuristic Suggestions
        tables = []
        if "." in s and HAS_BQ_CLIENT:
            try:
                billing_project = engine.url.host or os.getenv("GOOGLE_CLOUD_PROJECT")
                client = bigquery.Client(project=billing_project)
                bq_tables = client.list_tables(s)
                tables = [t.table_id for t in bq_tables]
            except Exception as e:
                click.echo(f"  ⚠️ Native listing failed: {e}. Falling back to Inspector.")
        
        if not tables:
            try:
                tables = inspector.get_table_names(schema=clean_dataset)
            except:
                try: tables = inspector.get_table_names(schema=s)
                except: tables = []
        
        click.echo(f"  📂 Found {len(tables)} tables to analyze.")
        sugg_keys = suggest_keys(inspector, s if '.' not in s else clean_dataset, tables)
        
        # Merge and Deduplicate
        if not sys_keys.is_empty() or not sugg_keys.is_empty():
            schema_df = pl.concat([sys_keys, sugg_keys], how="diagonal")
            schema_df = schema_df.unique(subset=["table_name", "column_name"], keep="first")
            all_dfs.append(schema_df.with_columns(pl.lit(s).alias("schema")))

    if not all_dfs:
        click.secho("❌ No keys identified or suggested.", fg='yellow')
        return

    final_df = pl.concat(all_dfs)
    
    # 3. Identify tables with missing keys
    tables_in_report = final_df.get_column("table_name").unique().to_list()
    # We need to know ALL tables to find which are missing
    all_tables_list = []
    for s in target_schemas:
        clean_dataset = s.split('.')[-1] if '.' in s else s
        try:
            t_names = inspector.get_table_names(schema=clean_dataset)
            all_tables_list.extend([{"schema": s, "table_name": t} for t in t_names if t not in tables_in_report])
        except: pass
    
    missing_keys_df = pl.DataFrame(all_tables_list)
    if not missing_keys_df.is_empty():
        missing_keys_df = missing_keys_df.with_columns([
            pl.lit("MISSING").alias("column_name"),
            pl.lit("❌ NO PK/FK FOUND").alias("constraint_type"),
            pl.lit(None).alias("ref_table"),
            pl.lit(None).alias("ref_column")
        ])
        final_df = pl.concat([final_df, missing_keys_df], how="diagonal")

    # Display Report
    click.secho("\n🔑 Key Identification Report", bold=True, underline=True)
    report_data = final_df.sort(["schema", "table_name"]).to_dicts()
    print(tabulate(report_data, headers="keys", tablefmt="simple_grid"))
    
    if not missing_keys_df.is_empty():
        click.secho(f"\n⚠️ Alert: {len(missing_keys_df)} tables are missing Primary or Foreign Keys!", fg='red', bold=True)
    
    # SQL Generation & Auto-Apply for BQ
    if engine.dialect.name == 'bigquery':
        suggestions = final_df.filter(pl.col("constraint_type").str.contains("SUGGESTED"))
        if not suggestions.is_empty():
            click.secho(f"\n💡 Found {len(suggestions)} key suggestions.", bold=True)
            
            action = questionary.select(
                "What would you like to do with these suggestions?",
                choices=[
                    "Just print the SQL",
                    "Apply them to the Database now (NOT ENFORCED)",
                    "Skip"
                ]
            ).ask()

            if not action or action == "Skip": return

            # Generate SQL List
            sql_statements = []
            for row in suggestions.to_dicts():
                s_name = row['schema']
                # Clean schema name for SQL if it has project
                clean_s = s_name.split('.')[-1] if '.' in s_name else s_name
                # But we need project.dataset if we are cross-project
                # Let's use the full path from the row if available, or reconstruct
                # Actually, the 'schema' in the row is the one we scanned
                
                t_name = row['table_name']
                c_name = row['column_name']
                ref_t = row['ref_table']
                ref_c = row['ref_column']
                
                if row['constraint_type'] == "SUGGESTED PK":
                    sql_statements.append(f"ALTER TABLE `{s_name}.{t_name}` ADD PRIMARY KEY(`{c_name}`) NOT ENFORCED")
                elif ref_t:
                    # Assume same schema for FK if not specified (simplification)
                    sql_statements.append(f"ALTER TABLE `{s_name}.{t_name}` ADD FOREIGN KEY(`{c_name}`) REFERENCES `{s_name}.{ref_t}`(`{ref_c}`) NOT ENFORCED")

            if action == "Just print the SQL":
                click.echo("\n-- BigQuery ADD CONSTRAINT Statements:")
                for s in sql_statements: click.secho(f"{s};", fg='green')
            
            elif action == "Apply them to the Database now (NOT ENFORCED)":
                click.echo(f"\n🚀 Applying {len(sql_statements)} constraints...")
                try:
                    with engine.begin() as conn:
                        for s in sql_statements:
                            try:
                                conn.execute(text(s))
                                click.secho(f" ✅ Applied: {s[:60]}...", fg='green')
                            except Exception as e:
                                click.secho(f" ❌ Failed: {str(e).splitlines()[0]}", fg='red')
                except Exception as e:
                    click.secho(f"❌ Transaction Failed: {e}", fg='red')

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is not None: return
    click.clear()
    click.secho("👋 Welcome to QualiDB AI!", fg='cyan', bold=True)
    click.secho("Your AI-powered Data Governance Assistant.\n", fg='cyan')
    choice = questionary.select("What would you like to do?", choices=[
        questionary.Choice("1. 🧠 Generate Documentation (AI)", value="generate"),
        questionary.Choice("2. 🔄 Refresh Schema from DB (Reverse Sync)", value="refresh"),
        questionary.Choice("3. 🔍 Run Data Quality Checks", value="qa"),
        questionary.Choice("4. 💾 Push Documentation to DB", value="push"),
        questionary.Choice("5. 🧹 Prune Schema (Remove Deleted)", value="prune"),
        questionary.Choice("6. ❌ Exit", value="exit")
    ]).ask()
    if choice == "exit" or not choice: click.echo("Goodbye! 👋"); sys.exit(0)
    if choice == "generate":
        model_choice = questionary.select("Select AI Model:", choices=["Claude 3.5 Sonnet (Recommended)", "Google Gemini 1.5 Flash (Fast/Free)", "GPT-3.5 Turbo"]).ask()
        model_map = {"Claude 3.5 Sonnet (Recommended)": "sonnet", "Google Gemini 1.5 Flash (Fast/Free)": "gemini", "GPT-3.5 Turbo": "gpt-3.5-turbo"}
        write_db = questionary.confirm("Write descriptions directly to Database?").ask()
        ctx.invoke(generate_schema, conn=None, output='models/schema.yml', model=model_map.get(model_choice, "sonnet"), write_db=write_db)
    elif choice == "refresh":
        ctx.invoke(refresh_schema, conn=None, input='models/schema.yml')
    elif choice == "qa": ctx.invoke(check_quality, conn=None)
    elif choice == "push":
        input_file = 'models/schema.yml'
        if questionary.confirm("Push ALL tables from schema file?").ask():
            ctx.invoke(push_to_db, conn=None, input=input_file)
        else:
            # Load and pick tables
            if os.path.exists(input_file):
                with open(input_file, 'r') as f: data = yaml.safe_load(f) or {}
                avail_tables = []
                for s in data.get('sources', []):
                    s_name = s.get('name')
                    for t in s.get('tables', []): avail_tables.append(f"{s_name}.{t['name']}")
                
                selected = questionary.checkbox("Select tables to push:", choices=sorted(avail_tables)).ask()
                if selected:
                    ctx.invoke(push_to_db, conn=None, input=input_file, target_tables=selected)
    elif choice == "prune":
         if questionary.confirm("This will remove missing tables/datasets from your YAML. Proceed?").ask():
            ctx.invoke(prune_schema, conn=None, input='models/schema.yml')

@click.command(name='generate-config')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
def generate_config(conn):
    if not conn: conn = click.prompt("Connection String")
    try: engine = validate_connection(conn); inspector = inspect(engine)
    except Exception as e: print(f"Connection Error: {e}"); return
    try: schemas = inspector.get_schema_names()
    except Exception as e: print(f"❌ Error fetching schemas: {e}"); return

    # --- 1. Select Schemas ---
    # We use sorted schemas directly. 
    # TIP: Press 'a' in the menu to toggle all, 'i' to invert.
    schema_choices = sorted(schemas)
    selected_schemas_input = questionary.checkbox(
        "Select Datasets/Schemas to scan:", 
        choices=schema_choices,
        instruction="(Space to select, 'a' to toggle all)"
    ).ask()
    
    if not selected_schemas_input: return
    target_schemas = selected_schemas_input

    # --- 2. Exclusion Filters ---
    click.echo("\n🚫 EXCLUSION FILTERS")
    click.echo("Enter words to ignore. (e.g. '_staging' will hide 'orders_staging')")
    exclude_input = click.prompt("Exclude patterns (comma-separated)", default="", show_default=False)
    exclude_patterns = [p.strip().lower() for p in exclude_input.split(',')] if exclude_input else []
    
    generated_checks = [] 

    # --- 3. Scan Tables & Generate Checks ---
    for schema in target_schemas:
        print(f"\n📂 Scanning Schema: {schema}...")
        try: 
            tables = inspector.get_table_names(schema=schema)
            views = inspector.get_view_names(schema=schema)
            all_items = tables + views
        except Exception: continue
        if not all_items: continue

        filtered_items = []
        for t in all_items:
            full_name = f"{schema}.{t}".lower()
            if not any(pat in t.lower() or pat in full_name for pat in exclude_patterns):
                filtered_items.append(t)
        if not filtered_items: continue

        # --- FIX: Direct Selection Only ---
        # We removed the manual "(Select All)" option to prevent logic errors.
        # The user selection is now taken literally.
        table_choices = sorted(filtered_items)
        final_selection = questionary.checkbox(
            f"Select tables/views in '{schema}':", 
            choices=table_choices,
            instruction=f"(Found {len(table_choices)} items. Space to select, 'a' to toggle all)"
        ).ask()

        if not final_selection: 
            print("   Skipping (not items selected).")
            continue

        print(f"   🤖 Analyzing {len(final_selection)} tables...")
        with click.progressbar(final_selection, label=f"   Processing {schema}") as bar:
            for table in bar:
                clean_table_name = table.split('.')[-1]
                try: col_names = [c['name'] for c in inspector.get_columns(clean_table_name, schema=schema)]
                except: continue

                # AI Suggestion 
                ai_suggestion = None
                try: ai_suggestion = get_ai_suggested_config("gemini-1.5-flash", clean_table_name, col_names)
                except: pass

                check_entry = {"table": f"{schema}.{clean_table_name}"}
                
                # Logic to find columns
                if ai_suggestion:
                    if ai_suggestion.get('freshness_col') in col_names: check_entry['freshness_col'] = ai_suggestion['freshness_col']
                    if ai_suggestion.get('completeness_col') in col_names: check_entry['completeness_col'] = ai_suggestion['completeness_col']
                
                if 'freshness_col' not in check_entry:
                    for c in col_names:
                        if any(x in c.lower() for x in ['ingested', 'updated', 'created', '_ts', '_date', 'timestamp']):
                            check_entry['freshness_col'] = c; break
                if 'completeness_col' not in check_entry:
                    for c in col_names:
                        if any(x in c.lower() for x in ['id', 'key', 'uuid', 'pk']):
                            check_entry['completeness_col'] = c; break
                
                generated_checks.append(check_entry)

    if not generated_checks: print("\n❌ No checks generated."); return
    
    # --- 4. Write to YAML ---
    output_file = "checks.yml"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_data = yaml.safe_load(f) or {}
                existing_tables = [c['table'] for c in existing_data.get('checks', [])]
                for new_check in generated_checks:
                    if new_check['table'] not in existing_tables:
                        if 'checks' not in existing_data: existing_data['checks'] = []
                        existing_data['checks'].append(new_check)
                generated_checks_yaml = existing_data.get('checks', [])
            except: generated_checks_yaml = generated_checks
    else:
        generated_checks_yaml = generated_checks
    
    with open(output_file, 'w') as f: yaml.dump({"version": 1.0, "checks": generated_checks_yaml}, f, sort_keys=False)
    print(f"\n✅ Configuration saved to: {os.path.abspath(output_file)}")

    # --- 5. Write to DB (Selective) ---
    click.secho(f"\n🚀 Generated configuration for {len(generated_checks)} tables.", bold=True)
    sync_choice = questionary.select(
        "Do you want to write these rules to the Database (table: `data_quality_rules`)?",
        choices=[
            "No (YAML only)",
            "Yes - Sync ALL",
            "Yes - Select Specific Tables"
        ]
    ).ask()

    if sync_choice == "No (YAML only)": return

    checks_to_sync = []
    if sync_choice == "Yes - Sync ALL":
        checks_to_sync = generated_checks
    elif sync_choice == "Yes - Select Specific Tables":
        avail_tables = [c['table'] for c in generated_checks]
        selected_db_tables = questionary.checkbox("Select tables to sync config for:", choices=avail_tables).ask()
        if not selected_db_tables: print("Skipping DB sync."); return
        checks_to_sync = [c for c in generated_checks if c['table'] in selected_db_tables]

    target_dataset = target_schemas[0] if target_schemas else "public"
    rules_table = f"{target_dataset}.data_quality_rules"
    if '.' not in rules_table and engine.dialect.name == 'bigquery':
         rules_table = f"`{target_dataset}.data_quality_rules`"

    create_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {rules_table} (
            rule_id STRING,
            created_at TIMESTAMP,
            table_name STRING,
            freshness_column STRING,
            completeness_column STRING,
            is_active BOOLEAN
        );
    """)

    insert_sql = text(f"""
        INSERT INTO {rules_table} 
        (rule_id, created_at, table_name, freshness_column, completeness_column, is_active)
        VALUES (:rid, CURRENT_TIMESTAMP(), :tbl, :fresh, :comp, TRUE)
    """)

    print(f"💾 Writing {len(checks_to_sync)} rules to {rules_table}...")
    try:
        with engine.begin() as conn:
            conn.execute(create_sql)
            for check in checks_to_sync:
                conn.execute(insert_sql, {
                    "rid": str(uuid.uuid4()),
                    "tbl": check['table'],
                    "fresh": check.get('freshness_col'),
                    "comp": check.get('completeness_col')
                })
        print("✅ Rules synced to Database successfully!")
    except Exception as e:
        print(f"❌ DB Sync Failed: {e}")


cli.add_command(generate_schema)
cli.add_command(check_quality)
cli.add_command(push_to_db)
cli.add_command(generate_config)
cli.add_command(prune_schema)
cli.add_command(refresh_schema)

if __name__ == '__main__':
    cli()