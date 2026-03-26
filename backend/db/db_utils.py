import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    user = os.getenv('DB_USER', 'jimbosmac')
    password = os.getenv('DB_PASS', '')
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'macrominds')

    if password:
        url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    else:
        url = f"postgresql://{user}@{host}:{port}/{db_name}"

    return create_engine(url)

def test_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                __import__('sqlalchemy').text("SELECT COUNT(*) FROM economic_data")
            )
            count = result.scalar()
            print(f"Connected to macrominds! Rows in economic_data: {count}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
