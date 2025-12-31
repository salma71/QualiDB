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
import uuid6  
from sqlalchemy import text
import json
warnings.filterwarnings("ignore", category=sa_exc.SAWarning) 
load_dotenv()

def save_results_to_db(engine, results, dataset_name):
    """
    Auto-creates the log table if missing, then inserts results.
    FIXED: Now properly qualifies the table name with the dataset.
    """
    run_id = str(uuid6.uuid7())
    
    # FIX: Ensure dataset_name is clean
    safe_dataset = dataset_name.replace("`", "")
    target_table = f"`{safe_dataset}.data_quality_logs`"

    # 1. Create Table SQL (BigQuery Syntax)
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
    
    # 2. Insert SQL
    insert_sql = text(f"""
        INSERT INTO {target_table} 
        (run_id, run_at, dataset_name, table_name, column_name, check_type, status, result_value)
        VALUES (:run_id, CURRENT_TIMESTAMP(), :dataset, :table, :col, :check, :status, :val)
    """)
    
    print(f"\n💾 Saving {len(results)} results to {target_table} (Run ID: {run_id})...")
    
    # Open a FRESH connection for the save operation
    try:
        with engine.begin() as conn:
            conn.execute(create_table_sql)
            for row in results:
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
    """
    Sends a cleaner, vertical layout Slack message.
    FIXED: No more truncation of long table names.
    """
    if not webhook_url: return

    fails = [r for r in results if "FAIL" in r[4] or "❌" in r[4]]
    warns = [r for r in results if "OLD" in r[4] or "⚠️" in r[4]]
    passes = [r for r in results if r not in fails and r not in warns]
    
    total_issues = len(fails) + len(warns)
    status_emoji = "🚨" if len(fails) > 0 else ("⚠️" if len(warns) > 0 else "✅")
    
    # Header
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{status_emoji} Data Quality Report", "emoji": True}
        },
        {"type": "divider"}
    ]

    def add_section(items, title, emoji):
        if not items: return
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{emoji} {title}*"}})
        
        # Use simple text blocks instead of 'fields' to allow full width for long names
        for row in items[:10]: # Limit to 10 to avoid spamming
            table_name = row[0].split('.')[-1] # Clean name slightly but keep it readable
            col_name = row[1]
            check_type = row[2]
            val = row[3]
            
            text_block = f"• *{table_name}*\n   ↳ {check_type} on `{col_name}`: *{val}*"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": text_block}
            })

    # Add sections
    if fails: add_section(fails, "Critical Failures", "❌")
    if warns: add_section(warns, "Warnings", "⚠️")
    
    if not fails and not warns:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"✅ *All {len(passes)} checks passed successfully.*"}
        })
    else:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"{len(passes)} checks passed."}]})

    try:
        requests.post(webhook_url, json={"blocks": blocks})
        print("🔔 Slack notification sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send Slack alert: {e}")
# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def get_secret(secret_id, project_id, version_id="latest"):
    """
    Fetches a secret value (like Slack Webhook) from Google Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    
    try:
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"❌ Failed to fetch secret '{secret_id}': {e}")
        return None
def get_bq_quoted_name(schema, table):
    if not schema: return f"`{table}`"
    parts = schema.split('.')
    quoted_parts = [f"`{p}`" for p in parts]
    quoted_parts.append(f"`{table}`")
    return ".".join(quoted_parts)

# In db_check.py, replace the clean_ai_response function:

def clean_ai_response(text):
    if not text: return ""
    # 1. Remove Markdown headers (allow leading whitespace)
    text = re.sub(r'^\s*#+ .*', '', text, flags=re.MULTILINE)
    
    # 2. Remove bold labels like "**Column:**" or "**Description**"
    # FIX: Changed regex to not require a newline (\n) at the end
    text = re.sub(r'\*\*.*?\*\*[:\s]*', '', text, flags=re.MULTILINE)
    
    # 3. Remove "Description:" prefix
    text = re.sub(r'^Description:\s*', '', text, flags=re.MULTILINE)
    
    return " ".join(text.split()).strip()

def update_db_description(connection, schema, table, column, description, dialect):
    if not description or "Error" in description: return
    try:
        safe_desc = description.replace("'", "''")
        if dialect == 'bigquery':
            safe_desc_bq = description.replace('"', '\\"')
            target = get_bq_quoted_name(schema, table)
            if column:
                sql = f'ALTER TABLE {target} ALTER COLUMN `{column}` SET OPTIONS(description="{safe_desc_bq}")'
            else:
                sql = f'ALTER TABLE {target} SET OPTIONS(description="{safe_desc_bq}")'
        elif dialect in ['postgresql', 'snowflake']:
            target = f"{schema}.{table}.{column}" if column else f"{schema}.{table}"
            obj_type = "COLUMN" if column else "TABLE"
            sql = f"COMMENT ON {obj_type} {target} IS '{safe_desc}'"
        else: return 
        connection.execute(text(sql))
    except Exception as e:
        click.secho(f"⚠️ Failed to write to DB ({table}.{column}): {e}", fg='yellow')

def get_inspector(connection_string):
    connection_string = connection_string.strip('/')
    try:
        engine = create_engine(connection_string)
        return inspect(engine), engine
    except Exception as e:
        click.secho(f"Error connecting: {e}", fg='red'); sys.exit(1)

# ==========================================
# 🧠 AI ENGINE
# ==========================================

def get_valid_google_model(api_key):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url)
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
            client = anthropic.Anthropic(api_key=api_key, timeout=20.0)
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
            if "not found" in last_error.lower():
                print(f"  > ⚠️  Alias failed, auto-discovering...", end="\r")
                try:
                    found_model = get_valid_claude_model(client, "sonnet")
                    msg = client.messages.create(model=found_model, max_tokens=100, messages=[{"role": "user", "content": prompt_text}])
                    return clean_ai_response(msg.content[0].text)
                except Exception as e: return f"Claude Auto-Discovery Failed: {repr(e)}"
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
# 🛠️ CHECK LOGIC
# ==========================================

def check_nulls(connection, schema, table, column, **kwargs):
    try:
        tbl_ref = get_bq_quoted_name(schema, table)
        query = text(f"SELECT COUNT(*) FROM {tbl_ref} WHERE {column} IS NULL")
        count = connection.execute(query).scalar()
        return ("PASS", "✅") if count == 0 else (f"FAIL ({count})", "Failed")
    except: return None

def check_uniqueness(connection, schema, table, column, **kwargs):
    try:
        tbl_ref = get_bq_quoted_name(schema, table)
        query = text(f"SELECT COUNT({column}) - COUNT(DISTINCT {column}) FROM {tbl_ref}")
        diff = connection.execute(query).scalar()
        return ("PASS", "✅") if diff == 0 else (f"{diff} duplicates", "Failed")
    except: return None

def check_freshness(connection, schema, table, column, **kwargs):
    col_lower = column.lower()
    dtype = str(kwargs.get('dtype', '')).lower()
    time_terms = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'registered', 'ingested', '_at', '_ts']
    if not any(t in dtype for t in ['date', 'time', 'timestamp', 'datetime']) and not any(x in col_lower for x in time_terms): return None 
    if 'status' in col_lower: return None

    try:
        tbl_ref = get_bq_quoted_name(schema, table)
        query = text(f"SELECT MAX({column}) FROM {tbl_ref}")
        last_update = connection.execute(query).scalar()
        if not last_update: return ("EMPTY", "⚪")
        
        # Handle String Timestamps
        if isinstance(last_update, str):
            try: 
                clean_ts = str(last_update).replace('Z', '+00:00')
                last_update = datetime.datetime.fromisoformat(clean_ts)
            except: return (f"Bad fmt: {str(last_update)[:10]}...", "⚠️")

        # FIX: Handle pure Date objects (convert to datetime at midnight)
        # We check `type` strictly or use isinstance but exclude datetime subclass
        if isinstance(last_update, datetime.date) and not isinstance(last_update, datetime.datetime):
             last_update = datetime.datetime.combine(last_update, datetime.datetime.min.time())

        # Handle Timezones
        if last_update.tzinfo:
            now = datetime.datetime.now(last_update.tzinfo)
        else:
            now = datetime.datetime.now()

        diff = now - last_update
        hours = diff.total_seconds() / 3600
        return (f"{hours:.1f}h ago", "✅") if hours < 24 else (f"{hours:.1f}h ago", "⚠️ OLD")
    except Exception as e:
        error_str = str(e)
        if "404" in error_str: return ("API 404 (Check SQL)", "❌")
        # Ensure we catch attribute errors too
        return (f"Err: {error_str[:30]}", "❌")

AVAILABLE_CHECKS = {"Nulls": check_nulls, "Freshness": check_freshness, "Unique": check_uniqueness}

# ==========================================
# 🖥️ CLI SUB-COMMANDS
# ==========================================

@click.command(name='generate-schema')
@click.option('--conn', prompt='Connection String')
@click.option('--output', default='models/schema.yml')
@click.option('--model', default='gpt-3.5-turbo')
@click.option('--write-db', is_flag=True, help="Write descriptions back to DB")
def generate_schema(conn, output, model, write_db):
    """Generates schema with Append/Merge Logic."""
    inspector, engine = get_inspector(conn)
    
    click.echo("Fetching datasets (schemas)...")
    schemas = inspector.get_schema_names()
    target_schema = questionary.select("Select Dataset/Schema:", choices=schemas).ask()
    if not target_schema: return

    click.echo("Fetching tables...")
    all_tables = inspector.get_table_names(schema=target_schema)
    if not all_tables: click.secho("No tables found!", fg='red'); return
    
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
                        update_db_description(c, target_schema, clean_table_name, None, final_t_desc, engine.dialect.name)
            
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
                            update_db_description(c, target_schema, clean_table_name, c_name, final_c_desc, engine.dialect.name)
                
                cols_data.append({"name": c_name, "description": final_c_desc, "data_type": str(col['type'])})
            current_tables_dict[table] = {"name": table, "description": final_t_desc, "columns": cols_data}

    target_source["tables"] = list(current_tables_dict.values())
    with open(output, 'w') as f: yaml.dump(schema_data, f, sort_keys=False)
    click.secho(f"\n✅ Finished. Merged into {output}", fg='green')

def truncate_name(name, length=25):
    """Shortens long table names for Slack mobile views."""
    if len(name) <= length: return name
    return name[:10] + "..." + name[-(length-13):]


@click.command(name='check-quality')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
def check_quality(conn):
    """Runs QA checks based on checks.yml configuration."""
    if not conn: conn = click.prompt("Connection String")
    
    try:
        engine = create_engine(conn)
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # 1. LOAD CONFIGURATION
    config_path = "checks.yml"
    if not os.path.exists(config_path):
        print("❌ No 'checks.yml' found. Please run 'generate-config' first.")
        return

    print(f"📂 Found configuration file: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    checks_to_run = config.get('checks', [])
    if not checks_to_run:
        print("⚠️ Config file is empty.")
        return

    print(f"Running {len(checks_to_run)} configured checks... 🏃")
    results = []

    # 2. RUN CHECKS
    default_schema = checks_to_run[0]['table'].split('.')[0] if '.' in checks_to_run[0]['table'] else "public"

    with engine.connect() as connection:
        with click.progressbar(checks_to_run, label="Scanning") as bar:
            for check in bar:
                full_table_name = check['table']
                
                # Parse Schema.Table
                if '.' in full_table_name:
                    schema_part, table_part = full_table_name.split('.', 1)
                else:
                    schema_part, table_part = default_schema, full_table_name

                # --- CHECK 1: FRESHNESS ---
                if 'freshness_col' in check:
                    col = check['freshness_col']
                    # SAFE EXECUTION: Check if result is None before unpacking
                    res = check_freshness(connection, schema_part, table_part, col)
                    
                    if res:
                        res_val, status = res
                        results.append([full_table_name, col, "Freshness", res_val, status])
                    else:
                        # Handle the error gracefully
                        results.append([full_table_name, col, "Freshness", "SQL Error", "Failed"])

                # --- CHECK 2: COMPLETENESS ---
                if 'completeness_col' in check:
                    col = check['completeness_col']
                    # SAFE EXECUTION: Check if result is None before unpacking
                    res = check_nulls(connection, schema_part, table_part, col)
                    
                    if res:
                        res_val, status = res
                        results.append([full_table_name, col, "Completeness", res_val, status])
                    else:
                        # Handle the error gracefully
                        results.append([full_table_name, col, "Completeness", "SQL Error", "Failed"])

    # 3. DISPLAY & NOTIFY
    print("\n" + tabulate(results, headers=["Table", "Column", "Check", "Result", "Status"], tablefmt="simple_grid"))

    # Save to DB?
    if click.confirm("\n💾 Save results to DB?"):
        save_results_to_db(engine, results, default_schema)

    # Slack Alert?
    if click.confirm("🔔 Send report to Slack?"):
        PROJECT_ID = "dosedaily-raw" 
        SLACK_SECRET_NAME = "SLACK_WEBHOOK_URL"
        
        print("🔑 Fetching Webhook...")
        webhook_url = get_secret(SLACK_SECRET_NAME, PROJECT_ID)
        
        if webhook_url:
            send_slack_alert(webhook_url, results, default_schema)
        else:
            print("❌ Could not fetch Slack Webhook.")

@click.command(name='push-to-db')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
@click.option('--input', default='models/schema.yml', help='Path to schema.yml')
def push_to_db(conn, input):
    """Reads schema.yml and writes descriptions back to the Database."""
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
                            update_db_description(connection, db_target_schema, final_table_name, None, table_desc, dialect)
                        bar.update(1)
                        for col in tbl.get('columns', []):
                            col_name = col['name']
                            col_desc = col.get('description')
                            if col_desc:
                                update_db_description(connection, db_target_schema, final_table_name, col_name, col_desc, dialect)
                            bar.update(1)
    click.secho(f"\n✅ Successfully pushed descriptions to database!", fg='green')

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """🚀 Enterprise DB Tool: AI Governance & Quality."""
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

def get_ai_suggested_config(model_name, table_name, columns):
    """Safely calls AI and handles bad JSON."""
    prompt = (
        f"I have a table '{table_name}' with columns: {columns}. "
        "Return a JSON object with keys 'freshness_col' (best timestamp) "
        "and 'completeness_col' (best ID/PK). Return null if none found. "
        "Return ONLY JSON. No markdown."
    )

    try:
        # Reuse your existing generation logic
        response_text = generate_ai_description(table_name, "config", model_name, context=prompt)
        
        if not response_text: return None

        # Clean Markdown (This is the fix for "Expecting value")
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_text)
    except Exception:
        return None  # Return None so the fallback logic takes over
        

@click.command(name='generate-config')
@click.option('--conn', envvar='DB_CONNECTION_STRING', help='DB Connection String')
def generate_config(conn):
    """Auto-generates checks.yml with Robust AI Handling & Fallbacks."""
    print("🚀 DEBUG: Version with JSON Fixes Loaded")

    if not conn: conn = click.prompt("Connection String")
    
    try:
        engine = create_engine(conn)
        inspector = inspect(engine)
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # ==========================================
    # STEP 1: SELECT SCHEMAS
    # ==========================================
    try:
        schemas = inspector.get_schema_names()
    except Exception as e:
        print(f"❌ Error fetching schemas: {e}")
        return

    schema_choices = ["(Select All)", Separator("----- Schemas -----")] + schemas
    
    selected_schemas_input = questionary.checkbox(
        "Select Datasets/Schemas to scan:", 
        choices=schema_choices
    ).ask()

    if not selected_schemas_input: return

    if "(Select All)" in selected_schemas_input:
        target_schemas = schemas
        print(f"\n🌍 Selected ALL {len(target_schemas)} schemas.")
    else:
        target_schemas = selected_schemas_input

    # ==========================================
    # STEP 2: DEFINE EXCLUSIONS
    # ==========================================
    click.echo("\n🚫 EXCLUSION FILTERS")
    click.echo("Enter words to ignore. (e.g. '_staging' will hide 'orders_staging')")
    exclude_input = click.prompt("Exclude patterns (comma-separated)", default="", show_default=False)
    
    exclude_patterns = [p.strip().lower() for p in exclude_input.split(',')] if exclude_input else []
    
    if exclude_patterns:
        print(f"   🛡️ Active Filters: {exclude_patterns}")

    # ==========================================
    # STEP 3: SCAN & GENERATE
    # ==========================================
    generated_checks = []
    
    for schema in target_schemas:
        print(f"\n📂 Scanning Schema: {schema}...")
        try:
            tables = inspector.get_table_names(schema=schema)
        except Exception as e:
            print(f"   ⚠️ Could not read tables from {schema}: {e}")
            continue

        if not tables: continue

        # --- FIX: ROBUST EXCLUSION FILTER ---
        filtered_tables = []
        for t in tables:
            full_name = f"{schema}.{t}".lower()
            table_name = t.lower()
            
            is_excluded = False
            for pattern in exclude_patterns:
                if pattern in table_name or pattern in full_name:
                    is_excluded = True
                    break
            
            if not is_excluded:
                filtered_tables.append(t)
        
        skipped_count = len(tables) - len(filtered_tables)
        if skipped_count > 0:
            print(f"   🗑️  Skipped {skipped_count} excluded tables.")

        if not filtered_tables:
            print("   ⚠️ All tables excluded by filter.")
            continue

        # --- SELECT ---
        table_choices = ["(Select All)", Separator("----- Tables -----")] + filtered_tables
        selected_input = questionary.checkbox(f"Select tables in '{schema}':", choices=table_choices).ask()
        
        if not selected_input: continue
        final_selection = filtered_tables if "(Select All)" in selected_input else selected_input

        # --- ANALYZE ---
        print(f"   🤖 Analyzing {len(final_selection)} tables...")
        
        with click.progressbar(final_selection, label=f"   Processing {schema}") as bar:
            for table in bar:
                clean_table_name = table.split('.')[-1]
                
                try: 
                    columns_info = inspector.get_columns(clean_table_name, schema=schema)
                    col_names = [c['name'] for c in columns_info]
                except Exception as e:
                    # Try fallback for BQ (sometimes passing schema fails)
                    try:
                        columns_info = inspector.get_columns(table)
                        col_names = [c['name'] for c in columns_info]
                    except:
                        continue

                # 1. Ask AI (With Safer Parsing)
                ai_suggestion = None
                try:
                    # Call AI (Make sure get_ai_suggested_config is defined!)
                    # If this function isn't defined, we skip straight to fallback
                    if 'get_ai_suggested_config' in globals():
                        ai_suggestion = get_ai_suggested_config("gemini-1.5-flash", clean_table_name, col_names)
                except Exception:
                    # If AI crashes, just silently ignore and use fallback
                    ai_suggestion = None

                check_entry = {
                    "table": f"{schema}.{clean_table_name}",
                }
                
                # 2. Apply AI Results
                if ai_suggestion:
                    if ai_suggestion.get('freshness_col') and ai_suggestion['freshness_col'] in col_names:
                        check_entry['freshness_col'] = ai_suggestion['freshness_col']
                    if ai_suggestion.get('completeness_col') and ai_suggestion['completeness_col'] in col_names:
                        check_entry['completeness_col'] = ai_suggestion['completeness_col']
                
                # 3. FIX: FALLBACK LOGIC (The "Plan B")
                
                # Fallback for Freshness (Time)
                if 'freshness_col' not in check_entry:
                    for c in col_names:
                        c_low = c.lower()
                        if any(x in c_low for x in ['ingested', 'updated', 'created', '_ts', '_date', 'timestamp', 'modified']):
                            check_entry['freshness_col'] = c
                            break
                
                # Fallback for Completeness (ID/Count)
                if 'completeness_col' not in check_entry:
                    for c in col_names:
                        c_low = c.lower()
                        if any(x in c_low for x in ['id', 'key', 'uuid', 'pk', 'sku', 'code', 'number']):
                            check_entry['completeness_col'] = c
                            break

                generated_checks.append(check_entry)

    # ==========================================
    # STEP 4: SAVE
    # ==========================================
    if not generated_checks:
        print("\n❌ No checks generated.")
        return

    output_file = "checks.yml"
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_data = yaml.safe_load(f) or {}
                existing_checks = existing_data.get('checks', [])
                if existing_checks:
                    generated_checks = existing_checks + generated_checks
            except: pass
    
    final_yaml = {"version": 1.0, "checks": generated_checks}
    
    with open(output_file, 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)
        
    print(f"\n✅ Configuration saved to: {os.path.abspath(output_file)}")

cli.add_command(generate_schema)
cli.add_command(check_quality)
cli.add_command(push_to_db)
cli.add_command(generate_config)

if __name__ == '__main__':
    cli()
    