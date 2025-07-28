import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(
        host='yamabiko.proxy.rlwy.net',     # or 'mysql-production-6397.up.railway.app'
        port=40693,                          # port from your Railway project
        user='root',                         # username from Railway
        password='CFqThCzDPlimUEMTRRPFgAOTOzaLOcpa',  # password from Railway
        database='SUBS_STAGING',   
        connection_timeout=5           # your database name
    )

    if connection.is_connected():
        print("✅ Connected to Railway MySQL database")
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print("📦 Tables in the database:")
        for table in tables:
            print(f" - {table[0]}")

except Error as e:
    print("❌ Error while connecting to MySQL:", e)

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("🔌 MySQL connection closed")
