"""
Database Module for AI-Based Dyscalculia Screening Tool
Implements SQLite schema matching Chapter 5.5 (E-R Diagram) and Chapter 4.8.
"""

import sqlite3
import hashlib
import os
import json
from datetime import datetime
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "dyscalculia_screener.db"


def get_connection():
    """Establish and return SQLite database connection with row factory."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Securely hash password using SHA-256 with cryptographic salt."""
    if not salt:
        salt = os.urandom(16).hex()
    hash_obj = hashlib.sha256((salt + password).encode('utf-8'))
    return hash_obj.hexdigest(), salt


def init_db():
    """Initialize database tables according to the E-R Diagram."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table (Chapter 5.5 USER entity)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'Student',
        age INTEGER DEFAULT 10,
        registration_date TEXT NOT NULL
    );
    """)

    # 2. Assessments Table (Chapter 5.5 ASSESSMENT entity)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assessments (
        assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        assessment_date TEXT NOT NULL,
        total_score REAL NOT NULL,
        max_score REAL NOT NULL,
        percentage REAL NOT NULL,
        total_time_seconds REAL NOT NULL,
        risk_level TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Completed',
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # 3. Responses Table (Chapter 5.5 RESPONSE entity)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        response_id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL,
        question_id INTEGER NOT NULL,
        category TEXT NOT NULL,
        submitted_answer TEXT,
        correct_answer TEXT,
        is_correct INTEGER NOT NULL,
        response_time_ms REAL NOT NULL,
        FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
    );
    """)

    # 4. Screening Results Table (Chapter 5.5 SCREENING_RESULT entity)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS screening_results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL,
        risk_outcome TEXT NOT NULL,
        risk_probability REAL NOT NULL,
        subitizing_acc REAL NOT NULL,
        magnitude_acc REAL NOT NULL,
        arithmetic_acc REAL NOT NULL,
        estimation_acc REAL NOT NULL,
        reasoning_acc REAL NOT NULL,
        avg_response_time_ms REAL NOT NULL,
        primary_deficit TEXT,
        recommendations TEXT,
        model_version TEXT NOT NULL DEFAULT 'RandomForest-1.0',
        result_date TEXT NOT NULL,
        FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()
    seed_default_users()


def seed_default_users():
    """Seed initial sample accounts for testing and demonstration."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM users")
    count = cursor.fetchone()['count']
    if count == 0:
        demo_accounts = [
            ("Aadarsh Dubey", "aadarsh@rizvi.edu", "student123", "Student", 12),
            ("Prof. Javed Pathan", "javed.pathan@rizvi.edu", "faculty123", "Educator", 42),
            ("Demo Student (Rohan)", "rohan@example.com", "rohan123", "Student", 10),
            ("Dr. Psychologist", "clinic@example.com", "clinic123", "Clinician", 38),
        ]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name, email, pwd, role, age in demo_accounts:
            pwd_hash, salt = hash_password(pwd)
            cursor.execute("""
            INSERT INTO users (name, email, password_hash, salt, role, age, registration_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, email, pwd_hash, salt, role, age, now))
        conn.commit()

        # Seed an initial sample assessment for demo student
        cursor.execute("SELECT user_id FROM users WHERE email='rohan@example.com'")
        rohan_user = cursor.fetchone()
        if rohan_user:
            uid = rohan_user['user_id']
            cursor.execute("""
            INSERT INTO assessments (user_id, assessment_date, total_score, max_score, percentage, total_time_seconds, risk_level, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (uid, "2026-10-25 14:30:00", 7.0, 15.0, 46.7, 185.4, "Moderate Risk", "Completed"))
            ass_id = cursor.lastrowid

            cursor.execute("""
            INSERT INTO screening_results (
                assessment_id, risk_outcome, risk_probability,
                subitizing_acc, magnitude_acc, arithmetic_acc, estimation_acc, reasoning_acc,
                avg_response_time_ms, primary_deficit, recommendations, model_version, result_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ass_id, "Moderate Risk", 64.5,
                66.7, 66.7, 33.3, 33.3, 33.3,
                2450.0, "Mental Arithmetic & Number Estimation",
                "Recommend multi-sensory manipulatives, Cuisenaire rods, and visual number line exercises.",
                "RandomForest-1.0", "2026-10-25 14:31:00"
            ))
            conn.commit()

    conn.close()


def register_user(name: str, email: str, password: str, role: str = "Student", age: int = 10) -> tuple[bool, str]:
    """Register a new user account with hashed password."""
    if not name.strip() or not email.strip() or not password.strip():
        return False, "All fields are required."
    
    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return False, "An account with this email already exists."

        pwd_hash, salt = hash_password(password)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO users (name, email, password_hash, salt, role, age, registration_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name.strip(), email, pwd_hash, salt, role, int(age), now))
        conn.commit()
        return True, "Registration successful! You can now log in."
    except Exception as e:
        return False, f"Registration error: {str(e)}"
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> dict | None:
    """Validate credentials and return user dictionary if valid."""
    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return None

    pwd_hash, _ = hash_password(password, user['salt'])
    if pwd_hash == user['password_hash']:
        return {
            'user_id': user['user_id'],
            'name': user['name'],
            'email': user['email'],
            'role': user['role'],
            'age': user['age'],
            'registration_date': user['registration_date']
        }
    return None


def get_user_by_id(user_id: int) -> dict | None:
    """Fetch user by primary key."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name, email, role, age, registration_date FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_assessment_record(user_id: int, total_score: float, max_score: float,
                           percentage: float, total_time_seconds: float,
                           risk_level: str, responses_list: list,
                           screening_result: dict) -> int:
    """
    Save complete assessment, all question responses, and AI screening outcome in a transaction.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO assessments (user_id, assessment_date, total_score, max_score, percentage, total_time_seconds, risk_level, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, now, total_score, max_score, percentage, total_time_seconds, risk_level, "Completed"))
        assessment_id = cursor.lastrowid

        # Insert individual question responses
        for resp in responses_list:
            cursor.execute("""
            INSERT INTO responses (assessment_id, question_id, category, submitted_answer, correct_answer, is_correct, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                assessment_id,
                resp.get('question_id', 0),
                resp.get('category', 'General'),
                str(resp.get('submitted_answer', '')),
                str(resp.get('correct_answer', '')),
                1 if resp.get('is_correct') else 0,
                float(resp.get('response_time_ms', 0.0))
            ))

        # Insert screening result
        cursor.execute("""
        INSERT INTO screening_results (
            assessment_id, risk_outcome, risk_probability,
            subitizing_acc, magnitude_acc, arithmetic_acc, estimation_acc, reasoning_acc,
            avg_response_time_ms, primary_deficit, recommendations, model_version, result_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment_id,
            screening_result.get('risk_outcome', risk_level),
            float(screening_result.get('risk_probability', 0.0)),
            float(screening_result.get('subitizing_acc', 0.0)),
            float(screening_result.get('magnitude_acc', 0.0)),
            float(screening_result.get('arithmetic_acc', 0.0)),
            float(screening_result.get('estimation_acc', 0.0)),
            float(screening_result.get('reasoning_acc', 0.0)),
            float(screening_result.get('avg_response_time_ms', 0.0)),
            screening_result.get('primary_deficit', 'None identified'),
            screening_result.get('recommendations', ''),
            screening_result.get('model_version', 'RandomForest-1.0'),
            now
        ))

        conn.commit()
        return assessment_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_user_assessments(user_id: int) -> list:
    """Retrieve all assessment records for a specific user ordered by latest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT a.*, r.risk_outcome, r.risk_probability, r.primary_deficit,
           r.subitizing_acc, r.magnitude_acc, r.arithmetic_acc, r.estimation_acc, r.reasoning_acc,
           r.avg_response_time_ms, r.recommendations
    FROM assessments a
    LEFT JOIN screening_results r ON a.assessment_id = r.assessment_id
    WHERE a.user_id = ?
    ORDER BY a.assessment_id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_assessment_details(assessment_id: int) -> dict | None:
    """Fetch complete assessment details including all responses."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT a.*, r.risk_outcome, r.risk_probability, r.primary_deficit,
           r.subitizing_acc, r.magnitude_acc, r.arithmetic_acc, r.estimation_acc, r.reasoning_acc,
           r.avg_response_time_ms, r.recommendations, u.name as user_name, u.age as user_age, u.email as user_email
    FROM assessments a
    JOIN users u ON a.user_id = u.user_id
    LEFT JOIN screening_results r ON a.assessment_id = r.assessment_id
    WHERE a.assessment_id = ?
    """, (assessment_id,))
    assessment = cursor.fetchone()

    if not assessment:
        conn.close()
        return None

    data = dict(assessment)

    cursor.execute("""
    SELECT * FROM responses WHERE assessment_id = ? ORDER BY response_id ASC
    """, (assessment_id,))
    responses = cursor.fetchall()
    data['responses'] = [dict(resp) for resp in responses]

    conn.close()
    return data


def get_system_stats() -> dict:
    """Calculate application-wide summary metrics for dashboards."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM users")
    total_users = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) as count, AVG(percentage) as avg_score, AVG(total_time_seconds) as avg_time FROM assessments")
    ass_row = cursor.fetchone()
    total_assessments = ass_row['count']
    avg_score = ass_row['avg_score'] or 0.0
    avg_time = ass_row['avg_time'] or 0.0

    cursor.execute("""
    SELECT risk_level, COUNT(*) as count FROM assessments GROUP BY risk_level
    """)
    risk_breakdown = {r['risk_level']: r['count'] for r in cursor.fetchall()}

    conn.close()
    return {
        'total_users': total_users,
        'total_assessments': total_assessments,
        'avg_score': round(avg_score, 1),
        'avg_time_sec': round(avg_time, 1),
        'risk_breakdown': risk_breakdown
    }


# Initialize DB upon import
init_db()
