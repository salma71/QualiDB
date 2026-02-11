import yaml
import re
import os

def extract_keys(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    print(f"🚀 Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    if not data or 'sources' not in data:
        print("❌ Invalid YAML structure or no sources found.")
        return

    pk_count = 0
    fk_count = 0

    # Regex patterns
    # PK: Match "Primary key" or "Uniquely identifies (this|each) record"
    pk_pattern = re.compile(r'Primary key|Uniquely identifies (?:this|each) record', re.IGNORECASE)
    # Match "Joins to TableName.columnName"
    fk_pattern = re.compile(r'Joins to ([A-Z][\w]+)\.([\w\d_]+)', re.IGNORECASE)

    for source in data.get('sources', []):
        for table in source.get('tables', []):
            columns = table.get('columns', [])
            for i, column in enumerate(columns):
                name = column.get('name', '')
                desc = column.get('description', '')
                if not desc:
                    desc = "" # ensure it's a string for regex

                # Check for Primary Key
                is_pk = False
                if pk_pattern.search(desc):
                    is_pk = True
                
                # Heuristic: If name is 'id' and it's the first column, or if it says "uniquely"
                if not is_pk and name.lower() == 'id' and (i == 0 or 'unique' in desc.lower()):
                    is_pk = True

                if is_pk:
                    column['primary_key'] = True
                    pk_count += 1

                # Check for Foreign Key
                fk_match = fk_pattern.search(desc)
                if fk_match:
                    ref_table = fk_match.group(1)
                    ref_col = fk_match.group(2)
                    column['foreign_key'] = f"{ref_table}.{ref_col}"
                    fk_count += 1

    print(f"✅ Extraction complete!")
    print(f"   • Primary Keys found: {pk_count}")
    print(f"   • Foreign Keys found: {fk_count}")

    with open(output_file, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    
    print(f"💾 Saved with keys to: {output_file}")

if __name__ == "__main__":
    extract_keys('models/skio_schema_fixed.yml', 'models/skio_schema_with_keys.yml')
