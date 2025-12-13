#!/usr/bin/env python3
import sqlite3
import csv
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

csv_path = Path('data/courses/courses.csv')
db_path = Path('data/courses/courses.db')

if not csv_path.exists():
    print('courses.csv not found at', csv_path)
    raise SystemExit(1)

print('Reading CSV...')
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f'Rows: {len(rows)}')

cols = [
    'course_title','course_code','abstract','description','teachers','ects','learning_outcomes',
    'semester','semester_start','semester_end','language','participants_max','location','campus',
    'level','course_type','department','programme_name','formal_prerequisites','prerequisites_recommended',
    'content','materials','literature','assessment_text','exam_type','offered_exchange','offered_guest','offered_single_subject'
]

# Create DB and table
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
cur.execute('''
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_title TEXT,
    course_code TEXT,
    abstract TEXT,
    description TEXT,
    teachers TEXT,
    ects REAL,
    learning_outcomes TEXT,
    semester TEXT,
    semester_start TEXT,
    semester_end TEXT,
    language TEXT,
    participants_max TEXT,
    location TEXT,
    campus TEXT,
    level TEXT,
    course_type TEXT,
    department TEXT,
    programme_name TEXT,
    formal_prerequisites TEXT,
    prerequisites_recommended TEXT,
    content TEXT,
    materials TEXT,
    literature TEXT,
    assessment_text TEXT,
    exam_type TEXT,
    offered_exchange TEXT,
    offered_guest TEXT,
    offered_single_subject TEXT
)
''')
conn.commit()

insert_cols = cols
placeholders = ','.join('?' for _ in insert_cols)
sql = f"INSERT INTO courses ({','.join(insert_cols)}) VALUES ({placeholders})"

rows_to_insert = []
for r in rows:
    vals = []
    for c in insert_cols:
        v = r.get(c)
        if v is None or (isinstance(v, str) and v.strip() == ''):
            vals.append(None)
            continue
        if c == 'ects':
            try:
                vals.append(float(v))
            except Exception:
                vals.append(None)
        else:
            vals.append(v)
    rows_to_insert.append(tuple(vals))

cur.executemany(sql, rows_to_insert)
conn.commit()
print(f'Inserted {len(rows_to_insert)} rows')
conn.close()
print('Database created at', db_path)
