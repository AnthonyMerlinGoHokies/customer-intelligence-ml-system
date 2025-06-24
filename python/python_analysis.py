# E-Commerce Analysis - From SQL Detective Work to Python Visualizations
# This takes your SQL discoveries and makes them VISUAL and IMPRESSIVE!

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set up beautiful plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

class ECommerceInsightsDashboard:
    def __init__(self):
        self.conn = sqlite3.connect('ecommerce.db')
        print("🔗 Connected to database")
        print("📊 Ready to create impressive visualizations!")
    
    def revenue_gap_analysis(self):
        """Visualize the £1.7M customer data gap you discovered"""
        
        print("\n💰 REVENUE GAP ANALYSIS")
        print("=" * 40)
        
        # Use your SQL discovery
        query = """
        SELECT 
            SUM(CASE WHEN CustomerID IS NULL THEN Quantity * UnitPrice ELSE 0 END) as no_customer_revenue,
            SUM(CASE WHEN CustomerID IS NOT NULL THEN Quantity * UnitPrice ELSE 0 END) as tracked_customer_revenue,
            SUM(Quantity * UnitPrice) as total_revenue,
            COUNT(CASE WHEN CustomerID IS NULL THEN 1 END) as untracked_transactions,
            COUNT(CASE WHEN CustomerID IS NOT NULL THEN 1 END) as tracked_transactions
        FROM transactions 
        WHERE Quantity > 0
        """
        
        result = pd.read_sql_query(query, self.conn)
        
        # Create impressive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Revenue Pie Chart
        revenue_data = [result['tracked_customer_revenue'].iloc[0], result['no_customer_revenue'].iloc[0]]
        labels = ['Tracked Customers\n£{:,.0f}'.format(revenue_data[0]), 
                 'Missing Customer Data\n£{:,.0f}'.format(revenue_data[1])]
        colors = ['#2E8B57', '#DC143C']
        
        wedges, texts, autotexts = ax1.pie(revenue_data, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, explode=(0, 0.1))
        ax1.set_title('💰 REVENUE GAP DISCOVERED\nTotal: £{:,.0f}'.format(result['total_revenue'].iloc[0]), 
                     fontsize=14, fontweight='bold')
        
        # 2. Transaction Count Comparison
        transaction_data = [result['tracked_transactions'].iloc[0], result['untracked_transactions'].iloc[0]]
        x_pos = ['Tracked\nCustomers', 'Missing\nCustomer Data']
        bars = ax2.bar(x_pos, transaction_data, color=['#2E8B57', '#DC143C'])
        ax2.set_title('📊 Transaction Volume Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Transactions')
        
        # Add value labels on bars
        for bar, value in zip(bars, transaction_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(transaction_data)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Business Impact Metrics
        missing_revenue = result['no_customer_revenue'].iloc[0]
        total_revenue = result['total_revenue'].iloc[0]
        missing_pct = (missing_revenue / total_revenue) * 100
        
        metrics = ['Missing Revenue', 'Missing %', 'Missing Transactions']
        values = [missing_revenue/1000000, missing_pct, result['untracked_transactions'].iloc[0]/1000]
        units = ['£M', '%', 'K Trans']
        
        bars = ax3.bar(metrics, values, color=['#FF6347', '#FF8C00', '#FF69B4'])
        ax3.set_title('🚨 Business Impact Metrics', fontsize=14, fontweight='bold')
        
        for i, (bar, value, unit) in enumerate(zip(bars, values, units)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.1f}{unit}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Potential Customer Acquisition
        # Estimate if we captured these customers
        avg_customer_value = result['tracked_customer_revenue'].iloc[0] / result['tracked_transactions'].iloc[0]
        potential_customers = result['untracked_transactions'].iloc[0]
        
        scenarios = ['Current\nTracked Revenue', 'Potential with\nAll Customers Tracked']
        current_revenue = result['tracked_customer_revenue'].iloc[0] / 1000000
        potential_revenue = result['total_revenue'].iloc[0] / 1000000
        
        bars = ax4.bar(scenarios, [current_revenue, potential_revenue], 
                      color=['#4169E1', '#32CD32'])
        ax4.set_title('🎯 Revenue Opportunity', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Revenue (£ Millions)')
        
        for bar, value in zip(bars, [current_revenue, potential_revenue]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max([current_revenue, potential_revenue])*0.01,
                    f'£{value:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('revenue_gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key insights
        print(f"💡 KEY BUSINESS INSIGHTS:")
        print(f"   • Missing customer data on £{missing_revenue:,.0f} ({missing_pct:.1f}% of revenue)")
        print(f"   • {result['untracked_transactions'].iloc[0]:,} transactions without customer tracking")
        print(f"   • Average transaction value: £{avg_customer_value:.2f}")
        print(f"   • Potential for improved customer analytics and marketing")
        
        return result
    
    def monthly_revenue_trends(self):
        """Show monthly trends including the missing customer data"""
        
        print("\n📈 MONTHLY REVENUE TRENDS")
        print("=" * 40)
        
        query = """
        SELECT 
            strftime('%Y-%m', InvoiceDate) as month,
            SUM(CASE WHEN CustomerID IS NULL THEN Quantity * UnitPrice ELSE 0 END) as untracked_revenue,
            SUM(CASE WHEN CustomerID IS NOT NULL THEN Quantity * UnitPrice ELSE 0 END) as tracked_revenue,
            SUM(Quantity * UnitPrice) as total_revenue
        FROM transactions 
        WHERE Quantity > 0
        GROUP BY strftime('%Y-%m', InvoiceDate)
        ORDER BY month
        """
        
        trends = pd.read_sql_query(query, self.conn)
        trends['month'] = pd.to_datetime(trends['month'])
        
        # Create stacked area chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Stacked area chart
        ax1.fill_between(trends['month'], 0, trends['tracked_revenue'], 
                        alpha=0.7, color='#2E8B57', label='Tracked Customer Revenue')
        ax1.fill_between(trends['month'], trends['tracked_revenue'], trends['total_revenue'], 
                        alpha=0.7, color='#DC143C', label='Missing Customer Data Revenue')
        
        ax1.set_title('📊 Monthly Revenue: Tracked vs Missing Customer Data', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Revenue (£)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Percentage of missing data over time
        trends['missing_pct'] = (trends['untracked_revenue'] / trends['total_revenue']) * 100
        
        ax2.plot(trends['month'], trends['missing_pct'], 
                marker='o', linewidth=3, markersize=8, color='#FF6347')
        ax2.set_title('🚨 Percentage of Revenue Missing Customer Data Over Time', 
                     fontsize=16, fontweight='bold')
        ax2.set_ylabel('Missing Customer Data (%)')
        ax2.set_xlabel('Month')
        ax2.grid(True, alpha=0.3)
        
        # Add average line
        avg_missing = trends['missing_pct'].mean()
        ax2.axhline(y=avg_missing, color='red', linestyle='--', alpha=0.7,
                   label=f'Average: {avg_missing:.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('monthly_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return trends
    
    def top_customers_analysis(self):
        """Analyze your best customers (the ones you CAN track!)"""
        
        print("\n👑 TOP CUSTOMERS ANALYSIS")
        print("=" * 40)
        
        query = """
        SELECT 
            CustomerID,
            SUM(Quantity * UnitPrice) as total_spent,
            COUNT(DISTINCT InvoiceNo) as total_orders,
            AVG(Quantity * UnitPrice) as avg_order_value,
            COUNT(DISTINCT StockCode) as unique_products,
            MIN(InvoiceDate) as first_purchase,
            MAX(InvoiceDate) as last_purchase
        FROM transactions 
        WHERE CustomerID IS NOT NULL AND Quantity > 0
        GROUP BY CustomerID
        ORDER BY total_spent DESC
        LIMIT 20
        """
        
        customers = pd.read_sql_query(query, self.conn)
        
        # Create customer analysis dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 10 customers by revenue
        top_10 = customers.head(10)
        bars = ax1.barh(range(len(top_10)), top_10['total_spent'])
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels([f'Customer {id}' for id in top_10['CustomerID']])
        ax1.set_xlabel('Total Spent (£)')
        ax1.set_title('💰 Top 10 Customers by Revenue', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Customer value distribution
        ax2.hist(customers['total_spent'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Total Spent (£)')
        ax2.set_ylabel('Number of Customers')
        ax2.set_title('💵 Customer Value Distribution', fontsize=14, fontweight='bold')
        ax2.axvline(customers['total_spent'].mean(), color='red', linestyle='--', 
                   label=f'Average: £{customers["total_spent"].mean():.0f}')
        ax2.legend()
        
        # Orders vs Spending relationship
        ax3.scatter(customers['total_orders'], customers['total_spent'], alpha=0.6)
        ax3.set_xlabel('Number of Orders')
        ax3.set_ylabel('Total Spent (£)')
        ax3.set_title('🛒 Orders vs Total Spending', fontsize=14, fontweight='bold')
        
        # Add correlation
        correlation = customers['total_orders'].corr(customers['total_spent'])
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Average order value distribution
        ax4.boxplot(customers['avg_order_value'])
        ax4.set_ylabel('Average Order Value (£)')
        ax4.set_title('📊 Average Order Value Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('customer_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return customers
    
    def create_executive_summary(self):
        """Create a final executive summary of your findings"""
        
        print("\n📋 EXECUTIVE SUMMARY - YOUR DATA ANALYSIS PROJECT")
        print("=" * 60)
        
        # Get summary statistics
        summary_query = """
        SELECT 
            COUNT(DISTINCT CustomerID) as tracked_customers,
            COUNT(DISTINCT InvoiceNo) as total_orders,
            SUM(Quantity * UnitPrice) as total_revenue,
            SUM(CASE WHEN CustomerID IS NULL THEN Quantity * UnitPrice ELSE 0 END) as missing_revenue,
            COUNT(CASE WHEN CustomerID IS NULL THEN 1 END) as missing_transactions
        FROM transactions 
        WHERE Quantity > 0
        """
        
        summary = pd.read_sql_query(summary_query, self.conn)
        
        missing_revenue = summary['missing_revenue'].iloc[0]
        total_revenue = summary['total_revenue'].iloc[0]
        missing_pct = (missing_revenue / total_revenue) * 100
        
        print(f"🎯 KEY FINDINGS:")
        print(f"   • Analyzed {total_revenue:,.0f} in total revenue")
        print(f"   • Discovered £{missing_revenue:,.0f} ({missing_pct:.1f}%) in untracked customer revenue")
        print(f"   • Identified {summary['missing_transactions'].iloc[0]:,} transactions without customer data")
        print(f"   • {summary['tracked_customers'].iloc[0]:,} customers successfully tracked")
        
        print(f"\n💡 BUSINESS RECOMMENDATIONS:")
        print(f"   1. Implement customer data capture for ALL transactions")
        print(f"   2. Investigate why {missing_pct:.1f}% of sales lack customer tracking")
        print(f"   3. Potential revenue uplift through better customer analytics")
        print(f"   4. Improve CRM and marketing capabilities")
        
        print(f"\n🏆 ANALYSIS ACHIEVEMENTS:")
        print(f"   ✅ Professional SQL data exploration")
        print(f"   ✅ Data quality assessment and cleaning")
        print(f"   ✅ Business impact quantification")
        print(f"   ✅ Advanced data visualization")
        print(f"   ✅ Executive-level insights and recommendations")
        
        print(f"\n📊 Files Created:")
        print(f"   • revenue_gap_analysis.png")
        print(f"   • monthly_trends_analysis.png") 
        print(f"   • customer_analysis.png")
        print(f"   • ecommerce.db (SQLite database)")

# Run the complete analysis
if __name__ == "__main__":
    print("🚀 TRANSFORMING YOUR SQL DISCOVERIES INTO IMPRESSIVE VISUALS!")
    print("=" * 60)
    
    dashboard = ECommerceInsightsDashboard()
    
    # Run all analyses
    revenue_results = dashboard.revenue_gap_analysis()
    trends_results = dashboard.monthly_revenue_trends()
    customer_results = dashboard.top_customers_analysis()
    dashboard.create_executive_summary()
    
    print("\n🎉 CONGRATULATIONS!")
    print("You've completed a professional-level data analysis project!")
    print("This demonstrates real data analyst skills that companies pay well for!")