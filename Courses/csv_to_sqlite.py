#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load course data from CSV into SQLite database for chatbot use.

This script:
1. Reads the courses.csv file
2. Creates a SQLite database
3. Creates a courses table with appropriate schema
4. Inserts all course data
"""

import csv
import sqlite3
from pathlib import Path

# Configuration
CSV_FILE = "Courses/output/courses.csv"
DB_FILE = "Courses/output/courses.db"

def create_table(cursor: sqlite3.Cursor):
    """Create the courses table with appropriate schema."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            course_title TEXT NOT NULL,
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
            participants_max INTEGER,
            location TEXT,
            campus TEXT,
            level TEXT,
            course_type TEXT,
            department TEXT,
            programme_name TEXT,
            schedule_or_programme TEXT,
            formal_prerequisites TEXT,
            prerequisites_recommended TEXT,
            content TEXT,
            materials TEXT,
            literature TEXT,
            assessment_text TEXT,
            assessment_items TEXT,
            exam_type TEXT,
            exam_variation TEXT,
            offered_guest TEXT,
            offered_exchange TEXT,
            offered_single_subject TEXT,
            price_eu_eea TEXT,
            links TEXT,
            emails TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for common search fields
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_course_code ON courses(course_code)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_course_title ON courses(course_title)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_semester ON courses(semester)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_level ON courses(level)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_programme ON courses(programme_name)
    """)

def load_data():
    """Load CSV data into SQLite database."""
    csv_path = Path(CSV_FILE)
    db_path = Path(DB_FILE)
    
    # Check if CSV exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create table
        create_table(cursor)
        
        # Clear existing data to avoid duplicates
        cursor.execute("DELETE FROM courses")
        print(f"Clearing existing data from database...")
        
        # Read and insert data from CSV
        with csv_path.open('r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Get fieldnames from CSV
            fieldnames = reader.fieldnames
            
            rows_inserted = 0
            for row in reader:
                # Prepare values for insertion
                values = [row.get(field) if row.get(field) else None for field in fieldnames]
                
                # Build INSERT statement
                placeholders = ','.join(['?'] * len(fieldnames))
                columns = ','.join(fieldnames)
                
                insert_sql = f"INSERT INTO courses ({columns}) VALUES ({placeholders})"
                
                cursor.execute(insert_sql, values)
                rows_inserted += 1
        
        # Commit the changes
        conn.commit()
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM courses")
        total_count = cursor.fetchone()[0]
        
        print(f"✓ Successfully loaded {rows_inserted} courses into {db_path}")
        print(f"✓ Total courses in database: {total_count}")
        
        # Show some statistics
        cursor.execute("SELECT COUNT(*) FROM courses WHERE offered_exchange = 'yes'")
        exchange_count = cursor.fetchone()[0]
        print(f"✓ Courses available for exchange students: {exchange_count}")
        
        cursor.execute("SELECT DISTINCT semester FROM courses WHERE semester IS NOT NULL ORDER BY semester")
        semesters = cursor.fetchall()
        if semesters:
            print(f"✓ Semesters available: {', '.join([s[0] for s in semesters])}")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    print("Loading course data into SQLite database...")
    print(f"CSV file: {CSV_FILE}")
    print(f"DB file: {DB_FILE}")
    print()
    
    try:
        load_data()
        print()
        print("Database created successfully! ✓")
        print(f"You can now use the database file: {DB_FILE}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

