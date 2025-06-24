SELECT 
    COUNT(*) as total_records,
    COUNT(CustomerID) as has_customer_id,
    COUNT(*) - COUNT(CustomerID) as missing_customer_id,
    COUNT(Description) as has_description,
    COUNT(*) - COUNT(Description) as missing_description,
    COUNT(UnitPrice) as has_unit_price,
    COUNT(*) - COUNT(UnitPrice) as missing_unit_price
FROM transactions;

SELECT 
    InvoiceNo,
    StockCode,
    Description,
    Quantity,
    UnitPrice,
    Country
FROM transactions 
WHERE CustomerID IS NULL
LIMIT 10;


SELECT 
    COUNT(DISTINCT InvoiceNo) as invoices_without_customer,
    COUNT(*) as total_items_without_customer,
    SUM(Quantity * UnitPrice) as revenue_without_customer
FROM transactions 
WHERE CustomerID IS NULL;

SELECT 
    SUM(CASE WHEN CustomerID IS NULL THEN Quantity * UnitPrice ELSE 0 END) as no_customer_revenue,
    SUM(CASE WHEN CustomerID IS NOT NULL THEN Quantity * UnitPrice ELSE 0 END) as tracked_customer_revenue,
    SUM(Quantity * UnitPrice) as total_revenue,
    ROUND(
        SUM(CASE WHEN CustomerID IS NULL THEN Quantity * UnitPrice ELSE 0 END) * 100.0 / 
        SUM(Quantity * UnitPrice), 2
    ) as pct_untracked_revenue
FROM transactions 
WHERE Quantity > 0;