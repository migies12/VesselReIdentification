import sqlite3

DB_PATH = "app_cache.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH, timeout=30)

def init_cloudy_table():
    with get_db_connection() as conn:
        # Optimizations to increase write speeds
        conn.execute("PRAGMA jounral_mode=WAL;")
        conn.execute("PRAGMA sybnchronous=OFF;")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cloudy_cache (
                event_id TEXT PRIMARY KEY,
                is_cloudy INTEGER
            )       
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS demo_events (
                event_id TEXT PRIMARY KEY         
            )           
        """)

def get_cached_cloudy_status(event_id):
    """
    True if image is stored as cloudy
    False if image stored as not cloudy
    None if image not in cache
    """
    with get_db_connection() as conn:
        result = conn.execute(
            "SELECT is_cloudy FROM cloudy_cache WHERE event_id = ?",
            (event_id,)
        ).fetchone()
        return bool(result[0]) if result else None
    
def cache_cloudy_status(event_id, is_cloudy):
    """
    Store whether an image is cloudy or not so we don't need to process it again
    """
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cloudy_cache (event_id, is_cloudy) VALUES (?, ?)",
            (event_id, 1 if is_cloudy else 0)
        )

def add_demo_event(event_id):
    """
    Add a new demo event to the demo events table
    """
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO demo_events (event_id) VALUES (?)",
            (event_id,)
        )

def remove_demo_event(event_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM demo_events WHERE event_id = ?", (event_id,))

def get_demo_events():
    """
    Get the list of event IDs to use for demo view
    """
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM demo_events")
        rows = cursor.fetchall()
        return [row[0] for row in rows]