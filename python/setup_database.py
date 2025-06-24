import sqlite3
import pandas as pd
import os

def setup_ecommerce_database():
    """Convert CSV to SQLite database"""
    
    # Path to your CSV file
    csv_path = '/Users/anthonymerlin/Desktop/SQL/E-commerce/customer-intelligence-project/data/data.csv'
    db_path = 'ecommerce.db'
    
    print("🔄 Loading CSV file...")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        print(f"✅ Loaded {len(df):,} rows from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Clean the data
    print("🧹 Cleaning data...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Create SQLite database
    print("🏗️ Creating SQLite database...")
    conn = sqlite3.connect(db_path)
    
    # Save to database
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # Create indexes for faster queries
    print("⚡ Creating indexes...")
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_customer ON transactions(CustomerID)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON transactions(InvoiceDate)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_country ON transactions(Country)')
    
    conn.close()
    print(f"✅ Database created: {db_path}")
    print("🎉 Setup complete! You can now run SQL queries.")

if __name__ == "__main__":
    setup_ecommerce_database()