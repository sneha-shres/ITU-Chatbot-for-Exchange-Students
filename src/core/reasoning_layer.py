"""
Reasoning Layer for Course Combination Queries
This module provides reasoning capabilities for complex queries that require
combinatorial logic, such as finding course combinations that sum to a specific
ECTS value.
"""

import re
import time
from typing import List, Dict, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# Keywords that indicate a combination/sum query
COMBINATION_KEYWORDS = [
    'combination', 'combinations', 'combine', 'sum', 'sums', 'sum up',
    'add up', 'total', 'together', 'pair', 'pairs', 'group', 'groups',
    'set', 'sets', 'which courses', 'what courses'
]


class CourseCombinationReasoner:
    
    def __init__(self, vector_db=None):
        self.vector_db = vector_db
        self._combo_cache = {}
    
    def is_combination_query(self, query: str) -> Tuple[bool, Optional[float]]:
        """Detect if a query is asking for course combinations that sum to ECTS.
        """
        ql = query.lower()
        print(ql)
        
        # Check for combination keywords
        has_combination_kw = any(kw in ql for kw in COMBINATION_KEYWORDS)
        print("has combination kw:", has_combination_kw)
        
        
        # Extract ECTS value from query
        ects_value = None
        # Pattern: "30 ects", "30 ECTS", "30.0 ects", etc.
        ects_patterns = [
            r'\b(\d{1,2}(?:[\.,]\d)?)\s*(?:ects?|points?)\b',
            r'(?:sum|total|add)\s+(?:up\s+)?(?:to\s+)?(\d{1,2}(?:[\.,]\d)?)\s*(?:ects?|points?)',
            r'(\d{1,2}(?:[\.,]\d)?)\s*(?:ects?|points?)\s+(?:in\s+)?(?:total|sum|together)'
        ]
        
        for pattern in ects_patterns:
            match = re.search(pattern, ql)
            if match:
                try:
                    raw = match.group(1).replace(',', '.')
                    ects_value = float(raw)
                    break
                except (ValueError, IndexError):
                    continue
        
        # If we found combination keywords and an ECTS value, it's a combination query
        if has_combination_kw and ects_value is not None:
            return True, ects_value
        
        # Also check for explicit "sum up to X" patterns
        if ects_value is not None:
            sum_patterns = [
                r'sum\s+(?:up\s+)?(?:to|of)',
                r'add\s+(?:up\s+)?(?:to|of)',
                r'total\s+(?:of|to)',
                r'combinations?\s+(?:that\s+)?(?:sum|add|total)'
            ]
            for pattern in sum_patterns:
                if re.search(pattern, ql):
                    return True, ects_value
        
        return False, None
    
    def find_combinations(
        self,
        target_ects: float,
        courses: List[Dict],
        max_combinations: Optional[int] = 10000,
        max_courses_per_combination: int = 5,
        tolerance: float = 0.0,
        timeout_seconds: Optional[float] = 3.0,
    ) -> List[Dict]:
        """Find all combinations of courses that sum to the target ECTS value.
        This solves a variant of the subset sum problem with constraints:
        - Find subsets of courses whose ECTS sum equals target_ects (within tolerance)
        - Limit the number of courses per combination
        - Limit the total number of combinations returned
        """

        if not courses:
            return []
        # Keep only courses with valid numeric ECTS (attempt robust conversion)
        valid_courses = []
        for c in courses:
            raw_ects = c.get('ects')
            if raw_ects is None:
                continue
            try:
                ects_val = float(raw_ects)
            except Exception:
                continue
            if ects_val > 0:
                # normalize stored ects to float
                c['ects'] = ects_val
                valid_courses.append(c)

        if not valid_courses:
            logger.warning("No courses with valid ECTS values found")
            return []

        scale = 10

        for c in valid_courses:
            # store scaled integer ects for algorithmic use
            try:
                c['_ects_scaled'] = int(round(float(c.get('ects', 0)) * scale))
            except Exception:
                c['_ects_scaled'] = None

        valid_courses = [c for c in valid_courses if c.get('_ects_scaled') is not None and c['_ects_scaled'] > 0]
        print(len(valid_courses))
        
        if not valid_courses:
            logger.warning("No courses with scaled ECTS values available")
            return []

        target_scaled = int(round(float(target_ects) * scale))

        # Build a stable identifier for caching (based on course codes/titles)
        codes = tuple(sorted([str(c.get('course_code') or c.get('course_title') or i) for i, c in enumerate(valid_courses)]))
        cache_key = (target_scaled, codes, max_courses_per_combination)
        if cache_key in self._combo_cache:
            return self._combo_cache[cache_key][:max_combinations] if max_combinations is not None else list(self._combo_cache[cache_key])

        combinations: List[Dict] = []
        
        if combinations and (max_combinations is None or len(combinations) >= max_combinations):
            self._combo_cache[cache_key] = combinations
            return combinations[:max_combinations] if max_combinations is not None else combinations

        # Prepare for optimized backtracking with memoization and timeout
        start_time = time.time()
        memo_failed: Set[Tuple[int, int]] = set()

        # Sort descending to try larger items first for better pruning
        valid_courses.sort(key=lambda x: x['_ects_scaled'], reverse=True)

        def backtrack(idx: int, current_combo: List[Dict], current_sum: int) -> bool:
            # timeout
            if timeout_seconds is not None and (time.time() - start_time) > timeout_seconds:
                return True

            if current_sum == target_scaled:
                total_ects = sum(float(x.get('ects', 0)) for x in current_combo)
                combinations.append({'courses': current_combo.copy(), 'total_ects': total_ects, 'course_count': len(current_combo)})
                if max_combinations is not None and len(combinations) >= max_combinations:
                    return True
                return False

            if current_sum > target_scaled:
                return False

            if len(current_combo) >= max_courses_per_combination:
                return False

            key = (idx, current_sum)
            if key in memo_failed:
                return False

            prev_count = len(combinations)
            for i in range(idx, len(valid_courses)):
                c = valid_courses[i]
                cs = c['_ects_scaled']
                if current_sum + cs > target_scaled:
                    continue
                current_combo.append(c)
                stop = backtrack(i + 1, current_combo, current_sum + cs)
                current_combo.pop()
                if stop:
                    return True

            if len(combinations) == prev_count:
                memo_failed.add(key)
            return False

        backtrack(0, [], 0)

        # cache and sort results for determinism
        combinations.sort(key=lambda x: (x['course_count'], abs(int(round(x['total_ects'] * scale)) - target_scaled)))
        print(f"Found {len(combinations)} combinations for target {target_ects} ECTS")
        
        # Deduplicate combinations that contain the same set of course codes/titles
        unique = []
        seen = set()
        for combo in combinations:
            # Build a stable key based on course codes when available, otherwise titles
            codes = []
            for c in combo.get('courses', []):
                code = (c.get('course_code') or '').strip()
                if not code:
                    code = (c.get('course_title') or '').strip()
                codes.append(code)
            key = tuple(sorted([s.lower() for s in codes if s]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(combo)

        self._combo_cache[cache_key] = unique
        return unique[:max_combinations] if max_combinations is not None else unique
    
    def get_all_courses_for_combinations(
        self,
        target_ects: float,
        filters: Optional[Dict] = None,
        limit: int = 200
    ) -> List[Dict]:
        """Get all courses from vector DB that could be used in combinations.
        
        This method searches the vector DB for course-related content and
        extracts structured course information (ECTS, title, code, etc.)
        """
        if not self.vector_db:
            logger.warning("Vector database not available for combination reasoning")
            return []
        
        try:
            # Search vector DB for course-related content
            search_query = "course ECTS credits"
            if filters:
                if filters.get('semester'):
                    search_query += f" {filters['semester']}"
                if filters.get('language'):
                    search_query += f" {filters['language']}"
            
            # Get more results to have enough candidates
            vector_results = self.vector_db.search(query=search_query, k=limit * 2)
            
            # Extract course information from vector results
            courses = []
            for result in vector_results:
                text = result.get('text', '')
                title = result.get('title', '')
                
                # Try to extract course information from the text
                course = self._extract_course_from_text(text, title, result)
                if course and course.get('ects'):
                    ects_val = float(course.get('ects', 0))
                    if 0 < ects_val <= target_ects:
                        # Apply filters
                        if self._matches_filters(course, filters):
                            courses.append(course)
            
            return courses[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving courses for combinations: {e}")
            return []

    def _extract_course_from_text(self, text: str, title: str, metadata: Dict) -> Optional[Dict]:
        course = {
            'course_title': title or '',
            'course_code': '',
            'ects': None,
            'semester': '',
            'language': '',
            'level': '',
            'offered_exchange': None,
            'description': text[:500]  # First 500 chars as description
        }
        
        # Try to extract ECTS (look for patterns like "7.5 ECTS", "15 ECTS", etc.)
        ects_match = re.search(r'(\d{1,2}(?:[\.,]\d)?)\s*(?:ECTS|ects|credits)', text, re.IGNORECASE)
        if ects_match:
            try:
                course['ects'] = float(ects_match.group(1).replace(',', '.'))
            except:
                pass
        
        # Try to extract course code (patterns like "BDSA", "KSADAPS1KU", etc.)
        code_match = re.search(r'\b([A-Z]{2,}[A-Z0-9]*)\b', text)
        if code_match:
            excluded = {'ECTS', 'IT', 'ITU', 'MSC', 'BSC', 'AI', 'ML', 'NLP', 'CV'}
            code = code_match.group(1)
            if code not in excluded:
                course['course_code'] = code
        
        # Extract semester if mentioned
        if 'autumn' in text.lower() or 'fall' in text.lower():
            course['semester'] = 'Autumn'
        elif 'spring' in text.lower():
            course['semester'] = 'Spring'
        
        # Extract language
        if 'english' in text.lower():
            course['language'] = 'English'
        elif 'danish' in text.lower() or 'dansk' in text.lower():
            course['language'] = 'Danish'
        
        # Only return if we found at least ECTS value
        if course['ects'] is not None:
            return course
        return None

    def _matches_filters(self, course: Dict, filters: Optional[Dict]) -> bool:
        """Check if a course matches the provided filters."""
        if not filters:
            return True
        
        if filters.get('language'):
            lang = (course.get('language') or '').lower()
            if filters['language'].lower() not in lang:
                return False
        
        if filters.get('semester'):
            sem = (course.get('semester') or '').lower()
            if filters['semester'].lower() not in sem:
                return False
        
        if filters.get('level'):
            lvl = (course.get('level') or '').lower()
            if filters['level'].lower() not in lvl:
                return False
        
        if filters.get('offered_exchange') is not None:
            oe = course.get('offered_exchange')
            if filters['offered_exchange']:
                if not oe or str(oe).lower() != 'yes':
                    return False
        
        return True
    
    def format_combinations_for_context(self, combinations: List[Dict]) -> str:
        """Format combination results into a context string for LLM.
        """
        if not combinations:
            return "No course combinations found that sum to the target ECTS value."

        import re

        def sanitize(s: str) -> str:
            if not s:
                return ''
            # remove common citation markers like [1], (1), 【1】 and superscript numbers
            s = re.sub(r"\[\s*\d+\s*\]", '', s)
            s = re.sub(r"\(\s*\d+\s*\)", '', s)
            s = re.sub(r"【\s*\d+\s*】", '', s)
            s = re.sub(r"\s*\^\d+", '', s)
            # collapse multiple spaces and trim
            s = re.sub(r"\s+", ' ', s).strip()
            # remove stray trailing punctuation like ' ;' or ' ,'
            s = re.sub(r"\s+[;,]+$", '', s)
            return s

        parts: List[str] = []
        parts.append(f"Found {len(combinations)} course combination(s) that sum to the target ECTS:\n")

        for i, combo in enumerate(combinations, 1):
            parts.append(f"\nCombination {i} ({combo.get('total_ects')} ECTS total, {combo.get('course_count')} courses):")
            for course in combo.get('courses', []):
                title = sanitize(course.get('course_title', 'Unknown'))
                code = sanitize(course.get('course_code', '') or '')
                ects = course.get('ects', 0)
                semester = sanitize(course.get('semester', '') or '')

                # Build course line without numeric prefix
                course_str = f"  {title}"
                if code:
                    course_str += f" ({code})"
                course_str += f" - {ects} ECTS"
                if semester:
                    course_str += f" - {semester}"
                lang = sanitize(course.get('language') or '')
                if lang:
                    course_str += f" - {lang}"
                oe = course.get('offered_exchange')
                if oe:
                    oe_s = sanitize(str(oe))
                    course_str += f" - Offered for exchange: {oe_s}"

                parts.append(course_str)

        out = "\n".join(parts)
        # Remove any lines that contain only citation markers like [1], (1), 【1】 or superscript numbers
        out = re.sub(r"(?m)^\s*(?:\[\s*\d+\s*\]|\(\s*\d+\s*\)|【\s*\d+\s*\】|\^\d+)\s*$\n?", "", out)
        return out
    
    def reason_about_query(
        self,
        query: str,
        target_ects: float,
        filters: Optional[Dict] = None
    ) -> Dict:
        """Main reasoning function: detect combination query and find solutions.
        """
        is_combo, detected_ects = self.is_combination_query(query)
        print(is_combo, detected_ects)
        
        
        # Use detected ECTS if available, otherwise use provided target_ects
        final_target = detected_ects if detected_ects is not None else target_ects
        
        if not is_combo or final_target is None:
            return {
                'is_combination_query': False,
                'target_ects': None,
                'combinations': [],
                'courses_used': [],
                'formatted_context': ''
            }
        
        # Get all relevant courses
        courses = self.get_all_courses_for_combinations(final_target, filters)

        
        
        if not courses:
            return {
                'is_combination_query': True,
                'target_ects': final_target,
                'combinations': [],
                'courses_used': [],
                'formatted_context': f"No courses found that could sum to {final_target} ECTS."
            }
        
        # Find combinations
        combinations = self.find_combinations(
            target_ects=final_target,
            courses=courses,
            max_combinations=5,  # return all combinations found
            max_courses_per_combination=5,
            tolerance=0.0  # Exact match only
        )
        
     
        
        # Format for context
        formatted = self.format_combinations_for_context(combinations)
        print(formatted)
        
        
        return {
            'is_combination_query': True,
            'target_ects': final_target,
            'combinations': combinations,
            'courses_used': courses,
            'formatted_context': formatted
        }