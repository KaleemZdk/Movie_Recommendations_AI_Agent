import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "movies.db"


def get_connection():
    """
    Create and return a SQLite database connection.
    """

    conn = sqlite3.connect(DB_PATH)

    # Allows SQLite to return rows like dictionaries
    conn.row_factory = sqlite3.Row

    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """
    Create all required tables if they don't already exist.
    """

    conn = get_connection()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # FAVORITES
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tmdb_id INTEGER NOT NULL UNIQUE,
            title TEXT NOT NULL,
            added_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # WATCHING
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watching (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tmdb_id INTEGER NOT NULL UNIQUE,
            title TEXT NOT NULL,
            added_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # REVIEWS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tmdb_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL,

            CHECK (rating >= 0 AND rating <= 10),

            UNIQUE(tmdb_id)
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# HELPER
# ============================================================

def current_time():
    """
    Return current UTC time in ISO format.
    """

    return datetime.now(timezone.utc).isoformat()


# ============================================================
# FAVORITES
# ============================================================

def add_favorite(tmdb_id: int, title: str):
    """
    Add a show/movie to favorites.

    Returns True if successfully added.
    Returns False if it already exists.
    """

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO favorites (tmdb_id, title, added_at)
            VALUES (?, ?, ?)
            """,
            (tmdb_id, title, current_time())
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


def list_favorites():
    """
    Return all favorite shows/movies.
    """

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT id, tmdb_id, title, added_at
        FROM favorites
        ORDER BY added_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def remove_favorite(tmdb_id: int):
    """
    Remove a show/movie from favorites.
    """

    conn = get_connection()

    cursor = conn.execute(
        """
        DELETE FROM favorites
        WHERE tmdb_id = ?
        """,
        (tmdb_id,)
    )

    conn.commit()

    deleted = cursor.rowcount > 0

    conn.close()

    return deleted


# ============================================================
# WATCHING
# ============================================================

def set_watching(tmdb_id: int, title: str):
    """
    Add a show/movie to the user's watching list.

    Returns True if successfully added.
    Returns False if it already exists.
    """

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO watching (tmdb_id, title, added_at)
            VALUES (?, ?, ?)
            """,
            (tmdb_id, title, current_time())
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


def list_watching():
    """
    Return all shows/movies currently being watched.
    """

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT id, tmdb_id, title, added_at
        FROM watching
        ORDER BY added_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def remove_watching(tmdb_id: int):
    """
    Remove a show/movie from the watching list.
    """

    conn = get_connection()

    cursor = conn.execute(
        """
        DELETE FROM watching
        WHERE tmdb_id = ?
        """,
        (tmdb_id,)
    )

    conn.commit()

    deleted = cursor.rowcount > 0

    conn.close()

    return deleted


# ============================================================
# REVIEWS
# ============================================================

def save_review(
    tmdb_id: int,
    rating: float,
    comment: str
):
    """
    Save or update a review.

    Rating must be between 0 and 10.
    """

    if not 0 <= rating <= 10:
        raise ValueError("Rating must be between 0 and 10.")

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO reviews (
            tmdb_id,
            rating,
            comment,
            created_at
        )
        VALUES (?, ?, ?, ?)

        ON CONFLICT(tmdb_id)
        DO UPDATE SET
            rating = excluded.rating,
            comment = excluded.comment,
            created_at = excluded.created_at
        """,
        (
            tmdb_id,
            rating,
            comment,
            current_time()
        )
    )

    conn.commit()
    conn.close()

    return True


def get_review(tmdb_id: int):
    """
    Get the review for a specific show/movie.
    """

    conn = get_connection()

    row = conn.execute(
        """
        SELECT
            id,
            tmdb_id,
            rating,
            comment,
            created_at
        FROM reviews
        WHERE tmdb_id = ?
        """,
        (tmdb_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return None

    return dict(row)


def list_reviews():
    """
    Return all reviews.
    """

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            id,
            tmdb_id,
            rating,
            comment,
            created_at
        FROM reviews
        ORDER BY created_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# ============================================================
# INITIALIZE DATABASE
# ============================================================

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")