"""
SQL Database Interface for Course Data

Provides functions to query the courses database for structured course information.
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CourseDatabase:
    """Interface for querying the courses SQLite database."""
    
    def __init__(self, db_path: str = None):
        """Initialize the course database connection.
        Args:
            db_path: Path to the courses SQLite database file. If None, resolve common locations.
        """
        from pathlib import Path
        # If caller provided a path, try that first
        if db_path:
            self.db_path = str(Path(db_path).expanduser())
        else:
            # Try a list of candidate paths relative to the repo and cwd
            repo_root = Path(__file__).parent.resolve()
            candidates = [
                repo_root / "Courses" / "output" / "courses.db",
                repo_root / "Courses" / "output" / "courses.sqlite",
                Path.cwd() / "Courses" / "output" / "courses.db",
                Path.cwd() / "courses.db",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            self.db_path = str(found) if found else None

        self._check_database_exists()
    
    def _check_database_exists(self):
        """Check if the database file exists."""
        if not self.db_path or not os.path.exists(self.db_path):
            logger.warning(f"Course database not found. Tried path: {self.db_path}")
            self.db_path = None
        else:
            logger.info(f"Using course DB at: {self.db_path}")
    
    def _connect(self) -> Optional[sqlite3.Connection]:
        """Create a database connection."""
        if not self.db_path or not os.path.exists(self.db_path):
            return None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def search_courses(
        self,
        query: str = None,
        course_code: str = None,
        course_title: str = None,
        semester: str = None,
        level: str = None,
        ects: float = None,
        offered_exchange: bool = None,
        programme: str = None,
        language: str = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict]:
        """Search for courses matching the given criteria.
        
        Args:
            query: General text search across title, description, abstract
            course_code: Exact course code match
            course_title: Partial title match
            semester: Semester filter (e.g., "Autumn 2026", "Spring 2026")
            level: Level filter (e.g., "BSc", "MSc")
            ects: ECTS credit points
            offered_exchange: Filter by exchange student availability
            programme: Programme name filter
            limit: Maximum number of results
            
        Returns:
            List of course dictionaries matching the criteria
        """
        if not self.db_path:
            return []
        
        conn = self._connect()
        if not conn:
            return []
        
        try:
            conditions = []
            params = []
            
            # Exact course code match
            if course_code:
                conditions.append("course_code = ?")
                params.append(course_code.upper())
            
            # Title search
            if course_title:
                conditions.append("course_title LIKE ?")
                params.append(f"%{course_title}%")
            
            # Semester filter
            if semester:
                conditions.append("semester = ?")
                params.append(semester)
            
            # Level filter
            if level:
                conditions.append("level = ?")
                params.append(level)

            # Language filter
            if language:
                conditions.append("language LIKE ?")
                params.append(f"%{language}%")
            
            # ECTS filter
            if ects:
                conditions.append("ects = ?")
                params.append(ects)
            
            # Exchange student availability
            if offered_exchange is not None:
                if offered_exchange:
                    conditions.append("offered_exchange = 'yes'")
                else:
                    conditions.append("(offered_exchange IS NULL OR offered_exchange != 'yes')")
            
            # Programme filter
            if programme:
                conditions.append("programme_name LIKE ?")
                params.append(f"%{programme}%")
            
            # General text search
            if query:
                query_conditions = [
                    "course_title LIKE ?",
                    "description LIKE ?",
                    "abstract LIKE ?",
                    "content LIKE ?",
                    "course_code LIKE ?"
                ]
                query_param = f"%{query}%"
                conditions.append(f"({' OR '.join(query_conditions)})")
                params.extend([query_param] * len(query_conditions))
            
            # Build SQL query
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            sql = f"""
                SELECT 
                    id, course_title, course_code, abstract, description,
                    teachers, ects, learning_outcomes, semester,
                    semester_start, semester_end, language, participants_max,
                    location, campus, level, course_type, department,
                    programme_name, formal_prerequisites, prerequisites_recommended,
                    content, materials, literature, assessment_text,
                    exam_type, offered_exchange, offered_guest, offered_single_subject
                FROM courses
                WHERE {where_clause}
                ORDER BY course_title
                LIMIT ? OFFSET ?
            """
            params.append(limit)
            params.append(offset)
            
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            courses = []
            for row in rows:
                course = dict(row)
                courses.append(course)
            
            return courses
            
        except Exception as e:
            logger.error(f"Error searching courses: {e}")
            return []
        finally:
            conn.close()
    
    def get_course_by_code(self, course_code: str) -> Optional[Dict]:
        """Get a specific course by its course code.
        
        Args:
            course_code: The course code (e.g., "BDSA", "DSYS")
            
        Returns:
            Course dictionary or None if not found
        """
        results = self.search_courses(course_code=course_code, limit=1)
        return results[0] if results else None

    def count_courses(
        self,
        query: str = None,
        course_code: str = None,
        course_title: str = None,
        semester: str = None,
        level: str = None,
        ects: float = None,
        offered_exchange: bool = None,
        programme: str = None,
        language: str = None
    ) -> int:
        """Return the total number of courses matching the given criteria."""
        if not self.db_path:
            return 0

        conn = self._connect()
        if not conn:
            return 0

        try:
            conditions = []
            params = []

            if course_code:
                conditions.append("course_code = ?")
                params.append(course_code.upper())

            if course_title:
                conditions.append("course_title LIKE ?")
                params.append(f"%{course_title}%")

            if semester:
                conditions.append("semester = ?")
                params.append(semester)

            if level:
                conditions.append("level = ?")
                params.append(level)

            if ects is not None:
                conditions.append("ects = ?")
                params.append(ects)

            if offered_exchange is not None:
                if offered_exchange:
                    conditions.append("offered_exchange = 'yes'")
                else:
                    conditions.append("(offered_exchange IS NULL OR offered_exchange != 'yes')")

            if programme:
                conditions.append("programme_name LIKE ?")
                params.append(f"%{programme}%")

            if language:
                conditions.append("language LIKE ?")
                params.append(f"%{language}%")

            if query:
                query_conditions = [
                    "course_title LIKE ?",
                    "description LIKE ?",
                    "abstract LIKE ?",
                    "content LIKE ?",
                    "course_code LIKE ?"
                ]
                query_param = f"%{query}%"
                conditions.append(f"({' OR '.join(query_conditions)})")
                params.extend([query_param] * len(query_conditions))

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            sql = f"SELECT COUNT(*) FROM courses WHERE {where_clause}"
            cursor = conn.execute(sql, params)
            count = cursor.fetchone()[0]
            return count
        except Exception:
            return 0
        finally:
            conn.close()
    
    def get_exchange_courses(self, semester: str = None, limit: int = 20) -> List[Dict]:
        """Get all courses available for exchange students.
        
        Args:
            semester: Optional semester filter
            limit: Maximum number of results
            
        Returns:
            List of courses available for exchange students
        """
        return self.search_courses(
            offered_exchange=True,
            semester=semester,
            limit=limit
        )
    
    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """Search courses using multiple keywords.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            
        Returns:
            List of matching courses
        """
        if not keywords:
            return []
        
        # Combine keywords into a search query
        query = " ".join(keywords)
        return self.search_courses(query=query, limit=limit)
    
    def get_courses_by_programme(self, programme: str, limit: int = 50) -> List[Dict]:
        """Get all courses for a specific programme.
        
        Args:
            programme: Programme name
            limit: Maximum number of results
            
        Returns:
            List of courses in the programme
        """
        return self.search_courses(programme=programme, limit=limit)
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the courses database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.db_path:
            return {"error": "Database not available"}
        
        conn = self._connect()
        if not conn:
            return {"error": "Could not connect to database"}
        
        try:
            stats = {}
            
            # Total courses
            cursor = conn.execute("SELECT COUNT(*) FROM courses")
            stats["total_courses"] = cursor.fetchone()[0]
            
            # Exchange courses
            cursor = conn.execute("SELECT COUNT(*) FROM courses WHERE offered_exchange = 'yes'")
            stats["exchange_courses"] = cursor.fetchone()[0]
            
            # Semesters
            cursor = conn.execute("SELECT DISTINCT semester FROM courses WHERE semester IS NOT NULL")
            stats["semesters"] = [row[0] for row in cursor.fetchall()]
            
            # Levels
            cursor = conn.execute("SELECT DISTINCT level FROM courses WHERE level IS NOT NULL")
            stats["levels"] = [row[0] for row in cursor.fetchall()]
            
            # Programmes
            cursor = conn.execute("SELECT DISTINCT programme_name FROM courses WHERE programme_name IS NOT NULL LIMIT 20")
            stats["programmes"] = [row[0] for row in cursor.fetchall()]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
        finally:
            conn.close()

