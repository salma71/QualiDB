import click
import yaml
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
        print(f"⚠️ Skipping {column_path}: 'google-cloud-bigquery' not installed.")
        return

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table}"
    
    try:
        bq_table = client.get_table(table_ref)
    except Exception as e:
        print(f"⚠️ Could not fetch table {table_ref}: {e}")
        return

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
        # print(f"   ✅ Updated nested schema: {column_path}")
    except Exception as e:
        print(f"   ❌ BQ Schema Update Failed ({column_path}): {e}")

def update_db_description(connection, schema, table, column, description, dialect, engine=None):
    if not description or "Error" in description: return
    
    safe_desc = description.replace("'", "''")
    
    try:
        # --- BIGQUERY SPECIAL HANDLING ---
        if dialect == 'bigquery':
            # Check if it's a nested column (contains dot)
            if column and "." in column:
                # Use Python Client API for nested fields
                project_id = engine.url.host or os.getenv("GOOGLE_CLOUD_PROJECT")
                if not project_id:
                    print(f"⚠️ Skipping {column}: Cannot determine Google Project ID.")
                    return
                
                # BigQuery schemas usually don't have dots in dataset names in the API
                # Clean the schema if it came in as "project.dataset"
                clean_dataset = schema.split('.')[-1]
                
                update_bq_nested_schema(project_id, clean_dataset, table, column, description)
                return

            # Standard Top-Level Column Update (Use SQL)
            safe_desc_bq = description.replace('"', '\\"')
            target = get_bq_quoted_name(schema, table)
            if column:
                sql = f'ALTER TABLE {target} ALTER COLUMN `{column}` SET OPTIONS(description="{safe_desc_bq}")'
            else:
                sql = f'ALTER TABLE {target} SET OPTIONS(description="{safe_desc_bq}")'
        
        # --- POSTGRES / SNOWFLAKE ---
        elif dialect in ['postgresql', 'snowflake']:
            target = f"{schema}.{table}.{column}" if column else f"{schema}.{table}"
            obj_type = "COLUMN" if column else "TABLE"
            sql = f"COMMENT ON {obj_type} {target} IS '{safe_desc}'"
        else: return 
        
        connection.execute(text(sql))
        
    except Exception as e:
        # Clean up error message for display
        err_msg = str(e).split('\n')[0]
        click.secho(f"⚠️ Failed to write to DB ({table}.{column}): {err_msg}", fg='yellow')

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
def generate_schema(conn, output, model, write_db):
    inspector, engine = get_inspector(conn)
    schemas = inspector.get_schema_names()
    target_schema = questionary.select("Select Dataset/Schema:", choices=schemas).ask()
    if not target_schema: return

    all_tables = inspector.get_table_names(schema=target_schema)
    selected_tables = questionary.checkbox("Select tables to document:", choices=["(Select All)"] + all_tables).ask()
    if not selected_tables: return
    final_tables = all_tables if "(Select All)" in selected_tables else selected_tables

    schema_data = {"version": 2, "sources": []}
    existing_desc_map = {}
    if os.path.exists(output):
        try:
            with open(output, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml: schema_data = loaded_yaml
                for src in schema_data.get('sources', []):
                    for t in src.get('tables', []):
                        t_name = t['name']
                        if 'description' in t: existing_desc_map[t_name] = t['description']
                        for c in t.get('columns', []):
                            existing_desc_map[f"{t_name}.{c['name']}"] = c.get('description')
        except: pass

    target_source = None
    for src in schema_data.get("sources", []):
        if src.get("name") == target_schema:
            target_source = src
            break
    if not target_source:
        target_source = {"name": target_schema, "tables": []}
        schema_data.setdefault("sources", []).append(target_source)

    current_tables_dict = {t['name']: t for t in target_source.get("tables", [])}
    overwrite_all = None 

    click.echo(f"\nProcessing {len(final_tables)} tables... ⏳")
    with click.progressbar(final_tables, label='Generating') as bar:
        for table in bar:
            clean_table_name = table.split('.')[-1] if '.' in table else table
            t_desc = existing_desc_map.get(table)
            should_gen_table = True
            
            if t_desc:
                if overwrite_all is True: should_gen_table = True
                elif overwrite_all is False: should_gen_table = False
                else:
                    print(f"\nExample found: Table '{table}' has desc: '{t_desc[:30]}...'")
                    choice = questionary.select("Overwrite?", choices=["Yes", "No", "Yes to All", "No to All"]).ask()
                    if choice == "Yes": should_gen_table = True
                    elif choice == "No": should_gen_table = False
                    elif choice == "Yes to All": should_gen_table = True; overwrite_all = True
                    elif choice == "No to All": should_gen_table = False; overwrite_all = False

            final_t_desc = t_desc
            if should_gen_table:
                final_t_desc = generate_ai_description(clean_table_name, "table", model)
                if write_db:
                    with engine.begin() as c: 
                        update_db_description(c, target_schema, clean_table_name, None, final_t_desc, engine.dialect.name, engine=engine)
            
            cols_data = []
            try: columns = inspector.get_columns(table, schema=target_schema)
            except: columns = inspector.get_columns(table)

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
                        with engine.begin() as c: 
                            update_db_description(c, target_schema, clean_table_name, c_name, final_c_desc, engine.dialect.name, engine=engine)
                cols_data.append({"name": c_name, "description": final_c_desc, "data_type": str(col['type'])})
            current_tables_dict[table] = {"name": table, "description": final_t_desc, "columns": cols_data}

    target_source["tables"] = list(current_tables_dict.values())
    with open(output, 'w') as f: yaml.dump(schema_data, f, sort_keys=False)
    click.secho(f"\n✅ Finished. Merged into {output}", fg='green')

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
def push_to_db(conn, input):
    if not conn: conn = click.prompt("Connection String")
    if not os.path.exists(input): click.secho(f"❌ File not found: {input}", fg='red'); return

    inspector, engine = get_inspector(conn)
    dialect = engine.dialect.name
    with open(input, 'r') as f: data = yaml.safe_load(f) or {}
    sources = data.get('sources', [])
    if not sources: click.echo("No sources found."); return

    total_items = 0
    for src in sources:
        for tbl in src.get('tables', []): total_items += 1 + len(tbl.get('columns', []))
    click.echo(f"Found {total_items} descriptions. Pushing to {dialect.upper()}... 🚀")

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
                        if table_desc:
                            update_db_description(connection, db_target_schema, final_table_name, None, table_desc, dialect, engine=engine)
                        bar.update(1)
                        for col in tbl.get('columns', []):
                            col_name = col['name']
                            col_desc = col.get('description')
                            if col_desc:
                                update_db_description(connection, db_target_schema, final_table_name, col_name, col_desc, dialect, engine=engine)
                            bar.update(1)
    click.secho(f"\n✅ Successfully pushed descriptions to database!", fg='green')

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is not None: return
    click.clear()
    click.secho("👋 Welcome to QualiDB AI!", fg='cyan', bold=True)
    click.secho("Your AI-powered Data Governance Assistant.\n", fg='cyan')
    choice = questionary.select("What would you like to do?", choices=[
        questionary.Choice("1. 🧠 Generate Documentation (AI)", value="generate"),
        questionary.Choice("2. 🔍 Run Data Quality Checks", value="qa"),
        questionary.Choice("3. 💾 Push Documentation to DB", value="push"),
        questionary.Choice("4. ❌ Exit", value="exit")
    ]).ask()
    if choice == "exit" or not choice: click.echo("Goodbye! 👋"); sys.exit(0)
    if choice == "generate":
        model_choice = questionary.select("Select AI Model:", choices=["Claude 3.5 Sonnet (Recommended)", "Google Gemini 1.5 Flash (Fast/Free)", "GPT-3.5 Turbo"]).ask()
        model_map = {"Claude 3.5 Sonnet (Recommended)": "sonnet", "Google Gemini 1.5 Flash (Fast/Free)": "gemini", "GPT-3.5 Turbo": "gpt-3.5-turbo"}
        write_db = questionary.confirm("Write descriptions directly to Database?").ask()
        ctx.invoke(generate_schema, conn=None, output='models/schema.yml', model=model_map.get(model_choice, "sonnet"), write_db=write_db)
    elif choice == "qa": ctx.invoke(check_quality, conn=None)
    elif choice == "push":
        if questionary.confirm("Proceed?").ask(): ctx.invoke(push_to_db, conn=None, input='models/schema.yml')

@click.command(name='generate-config')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
def generate_config(conn):
    if not conn: conn = click.prompt("Connection String")
    try: engine = validate_connection(conn); inspector = inspect(engine)
    except Exception as e: print(f"Connection Error: {e}"); return
    try: schemas = inspector.get_schema_names()
    except Exception as e: print(f"❌ Error fetching schemas: {e}"); return

    schema_choices = ["(Select All)", Separator("----- Schemas -----")] + schemas
    selected_schemas_input = questionary.checkbox("Select Datasets/Schemas to scan:", choices=schema_choices).ask()
    if not selected_schemas_input: return
    target_schemas = schemas if "(Select All)" in selected_schemas_input else selected_schemas_input

    click.echo("\n🚫 EXCLUSION FILTERS")
    click.echo("Enter words to ignore. (e.g. '_staging' will hide 'orders_staging')")
    exclude_input = click.prompt("Exclude patterns (comma-separated)", default="", show_default=False)
    exclude_patterns = [p.strip().lower() for p in exclude_input.split(',')] if exclude_input else []
    
    generated_checks = []
    for schema in target_schemas:
        print(f"\n📂 Scanning Schema: {schema}...")
        try: tables = inspector.get_table_names(schema=schema)
        except Exception: continue
        if not tables: continue

        filtered_tables = []
        for t in tables:
            full_name = f"{schema}.{t}".lower()
            if not any(pat in t.lower() or pat in full_name for pat in exclude_patterns):
                filtered_tables.append(t)
        if not filtered_tables: continue

        table_choices = ["(Select All)", Separator("----- Tables -----")] + filtered_tables
        selected_input = questionary.checkbox(f"Select tables in '{schema}':", choices=table_choices).ask()
        if not selected_input: continue
        final_selection = filtered_tables if "(Select All)" in selected_input else selected_input

        print(f"   🤖 Analyzing {len(final_selection)} tables...")
        with click.progressbar(final_selection, label=f"   Processing {schema}") as bar:
            for table in bar:
                clean_table_name = table.split('.')[-1]
                try: col_names = [c['name'] for c in inspector.get_columns(clean_table_name, schema=schema)]
                except: continue

                ai_suggestion = None
                try: ai_suggestion = get_ai_suggested_config("gemini-1.5-flash", clean_table_name, col_names)
                except: pass

                check_entry = {"table": f"{schema}.{clean_table_name}"}
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
    output_file = "checks.yml"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_data = yaml.safe_load(f) or {}
                if existing_data.get('checks'): generated_checks = existing_data.get('checks') + generated_checks
            except: pass
    
    with open(output_file, 'w') as f: yaml.dump({"version": 1.0, "checks": generated_checks}, f, sort_keys=False)
    print(f"\n✅ Configuration saved to: {os.path.abspath(output_file)}")

cli.add_command(generate_schema)
cli.add_command(check_quality)
cli.add_command(push_to_db)
cli.add_command(generate_config)

if __name__ == '__main__':
    cli()