import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT,
    role TEXT
)
""")

# Patients table
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    risk TEXT,
    department TEXT,
    confidence REAL
)
""")

# Insert demo users
cursor.execute("INSERT INTO users (username,password,role) VALUES ('patient1','1234','patient')")
cursor.execute("INSERT INTO users (username,password,role) VALUES ('doctor1','1234','doctor')")

conn.commit()
conn.close()

print("Database Created Successfully")