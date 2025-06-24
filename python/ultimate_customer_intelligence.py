# ULTIMATE ML PROJECT: CHURN PREDICTION + PRODUCT RECOMMENDATIONS
# This combines both ML models for maximum business impact!

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class UltimateCustomerIntelligence:
    def __init__(self):
        self.conn = sqlite3.connect('../ecommerce.db')
        self.churn_model = None
        self.feature_importance = None
        self.customer_data = None
        self.product_similarity = None
        
        print("🤖 ULTIMATE CUSTOMER INTELLIGENCE SYSTEM")
        print("=" * 50)
        print("🎯 Phase 1: Churn Prediction")
        print("🛒 Phase 2: Product Recommendations")
        print("💰 Goal: Save customers + Increase revenue")
        
    def create_customer_features(self):
        """Phase 1: Engineer features for churn prediction"""
        
        print("\n🔧 ENGINEERING FEATURES FOR CHURN PREDICTION...")
        
        # Get analysis date
        analysis_date_query = "SELECT MAX(InvoiceDate) FROM transactions"
        last_date = pd.read_sql_query(analysis_date_query, self.conn).iloc[0, 0]
        analysis_date = pd.to_datetime(last_date) + timedelta(days=1)
        
        print(f"📅 Analysis Date: {analysis_date.date()}")
        
        # Calculate comprehensive customer features
        customer_query = """
        SELECT 
            CustomerID,
            COUNT(DISTINCT InvoiceNo) as total_orders,
            SUM(Quantity * UnitPrice) as total_spent,
            AVG(Quantity * UnitPrice) as avg_order_value,
            COUNT(DISTINCT StockCode) as unique_products,
            SUM(Quantity) as total_items,
            COUNT(DISTINCT strftime('%Y-%m', InvoiceDate)) as active_months,
            JULIANDAY(MAX(InvoiceDate)) - JULIANDAY(MIN(InvoiceDate)) as customer_lifespan_days,
            MAX(InvoiceDate) as last_purchase_date,
            MIN(InvoiceDate) as first_purchase_date,
            COUNT(DISTINCT Country) as countries_purchased_from
        FROM transactions 
        WHERE CustomerID IS NOT NULL AND Quantity > 0
        GROUP BY CustomerID
        """
        
        customer_data = pd.read_sql_query(customer_query, self.conn)
        
        # Calculate recency
        customer_data['last_purchase_date'] = pd.to_datetime(customer_data['last_purchase_date'])
        customer_data['first_purchase_date'] = pd.to_datetime(customer_data['first_purchase_date'])
        customer_data['recency_days'] = (analysis_date - customer_data['last_purchase_date']).dt.days
        
        # Calculate additional features
        customer_data['order_frequency'] = customer_data['total_orders'] / (customer_data['customer_lifespan_days'] + 1)
        customer_data['avg_monthly_orders'] = customer_data['total_orders'] / (customer_data['active_months'] + 1)
        customer_data['avg_items_per_order'] = customer_data['total_items'] / customer_data['total_orders']
        customer_data['spending_velocity'] = customer_data['total_spent'] / (customer_data['customer_lifespan_days'] + 1)
        
        # Define churn (no purchase in 60 days)
        churn_threshold = 60
        customer_data['is_churned'] = (customer_data['recency_days'] > churn_threshold).astype(int)
        
        # Add customer segments based on value
        customer_data['value_segment'] = pd.qcut(
            customer_data['total_spent'], 
            q=4, 
            labels=['Low Value', 'Medium Value', 'High Value', 'VIP']
        )
        
        print(f"✅ Feature Engineering Complete!")
        print(f"   📊 Total customers: {len(customer_data):,}")
        print(f"   🚨 Churned customers: {customer_data['is_churned'].sum():,}")
        print(f"   📈 Churn rate: {customer_data['is_churned'].mean()*100:.1f}%")
        print(f"   💰 Average customer value: £{customer_data['total_spent'].mean():.0f}")
        
        self.customer_data = customer_data
        return customer_data
    
    def train_churn_model(self):
        """Phase 1: Train ML model to predict churn"""
        
        print("\n🤖 TRAINING CHURN PREDICTION MODEL...")
        
        # Select features for the model
        feature_columns = [
            'total_orders', 'total_spent', 'avg_order_value', 'unique_products',
            'total_items', 'active_months', 'customer_lifespan_days', 'recency_days',
            'order_frequency', 'avg_monthly_orders', 'avg_items_per_order', 
            'spending_velocity', 'countries_purchased_from'
        ]
        
        X = self.customer_data[feature_columns]
        y = self.customer_data['is_churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.churn_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.churn_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_proba = self.churn_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.churn_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"🎯 MODEL PERFORMANCE:")
        print(f"   🏆 AUC Score: {auc_score:.3f}")
        print(f"   📚 Training samples: {len(X_train):,}")
        print(f"   🧪 Test samples: {len(X_test):,}")
        
        print(f"\n🔍 TOP CHURN PREDICTORS:")
        for _, row in self.feature_importance.head(5).iterrows():
            print(f"   📊 {row['feature']}: {row['importance']:.3f}")
        
        # Predict churn probability for all customers
        all_probs = self.churn_model.predict_proba(X)[:, 1]
        self.customer_data['churn_probability'] = all_probs
        
        # Create risk segments
        self.customer_data['risk_segment'] = pd.cut(
            all_probs,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        return auc_score
    
    def build_recommendation_engine(self):
        """Phase 2: Build product recommendation system"""
        
        print("\n🛒 BUILDING PRODUCT RECOMMENDATION ENGINE...")
        
        # Get product purchase data
        product_query = """
        SELECT 
            CustomerID,
            StockCode,
            Description,
            SUM(Quantity) as quantity_bought,
            SUM(Quantity * UnitPrice) as revenue_from_product,
            COUNT(DISTINCT InvoiceNo) as times_purchased
        FROM transactions 
        WHERE CustomerID IS NOT NULL AND Quantity > 0 AND Description IS NOT NULL
        GROUP BY CustomerID, StockCode, Description
        """
        
        product_data = pd.read_sql_query(product_query, self.conn)
        
        # Create customer-product matrix
        customer_product_matrix = product_data.pivot_table(
            index='CustomerID',
            columns='StockCode', 
            values='quantity_bought',
            fill_value=0
        )
        
        # Calculate product similarity using cosine similarity
        product_similarity = cosine_similarity(customer_product_matrix.T)
        self.product_similarity = pd.DataFrame(
            product_similarity,
            index=customer_product_matrix.columns,
            columns=customer_product_matrix.columns
        )
        
        # Get product info for recommendations
        product_info_query = """
        SELECT 
            StockCode,
            Description,
            AVG(UnitPrice) as avg_price,
            SUM(Quantity) as total_sold,
            COUNT(DISTINCT CustomerID) as customers_bought
        FROM transactions 
        WHERE Quantity > 0 AND Description IS NOT NULL
        GROUP BY StockCode, Description
        ORDER BY total_sold DESC
        """
        
        self.product_info = pd.read_sql_query(product_info_query, self.conn)
        self.customer_product_matrix = customer_product_matrix
        
        print(f"✅ Recommendation Engine Built!")
        print(f"   🛍️  Products analyzed: {len(customer_product_matrix.columns):,}")
        print(f"   👥 Customers analyzed: {len(customer_product_matrix.index):,}")
        print(f"   📊 Total interactions: {(customer_product_matrix > 0).sum().sum():,}")
        
        return product_data
    
    def get_recommendations_for_customer(self, customer_id, n_recommendations=5):
        """Get product recommendations for a specific customer"""
        
        if customer_id not in self.customer_product_matrix.index:
            return None
        
        # Get products this customer has bought
        customer_purchases = self.customer_product_matrix.loc[customer_id]
        purchased_products = customer_purchases[customer_purchases > 0].index
        
        # Find similar products
        recommendations = {}
        
        for product in purchased_products:
            if product in self.product_similarity.index:
                similar_products = self.product_similarity[product].sort_values(ascending=False)
                
                # Add similar products that customer hasn't bought
                for similar_product, similarity in similar_products.items():
                    if (similar_product not in purchased_products and 
                        similar_product not in recommendations):
                        recommendations[similar_product] = similarity
        
        # Sort and get top recommendations
        top_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Get product details
        rec_details = []
        for product_code, similarity in top_recs:
            product_info = self.product_info[self.product_info['StockCode'] == product_code]
            if not product_info.empty:
                rec_details.append({
                    'product_code': product_code,
                    'description': product_info.iloc[0]['Description'],
                    'avg_price': product_info.iloc[0]['avg_price'],
                    'popularity': product_info.iloc[0]['total_sold'],
                    'similarity_score': similarity
                })
        
        return rec_details
    
    def create_ultimate_dashboard(self):
        """Create comprehensive dashboard combining both models"""
        
        print("\n📊 CREATING ULTIMATE CUSTOMER INTELLIGENCE DASHBOARD...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Churn Risk Distribution (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        risk_counts = self.customer_data['risk_segment'].value_counts()
        colors = ['#2E8B57', '#FF8C00', '#DC143C']
        wedges, texts, autotexts = ax1.pie(
            risk_counts.values, 
            labels=risk_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            explode=(0, 0, 0.1)
        )
        ax1.set_title('🚨 Customer Risk Distribution', fontsize=14, fontweight='bold')
        
        # 2. Revenue at Risk (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        revenue_by_risk = self.customer_data.groupby('risk_segment')['total_spent'].sum()
        bars = ax2.bar(revenue_by_risk.index, revenue_by_risk.values, color=colors)
        ax2.set_title('💰 Revenue at Risk', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Revenue (£)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, revenue_by_risk.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(revenue_by_risk)*0.01,
                    f'£{value:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Feature Importance (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        top_features = self.feature_importance.head(8)
        bars = ax3.barh(range(len(top_features)), top_features['importance'])
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels([feat.replace('_', ' ').title() for feat in top_features['feature']], fontsize=10)
        ax3.set_xlabel('Importance')
        ax3.set_title('🔍 Top Churn Predictors', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        
        # 4. Churn by Value Segment (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        churn_by_value = self.customer_data.groupby('value_segment')['is_churned'].mean() * 100
        bars = ax4.bar(churn_by_value.index, churn_by_value.values, color='coral')
        ax4.set_title('📊 Churn Rate by Customer Value', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Churn Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, churn_by_value.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(churn_by_value)*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Customer Value vs Churn Probability (Middle Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(
            self.customer_data['total_spent'], 
            self.customer_data['churn_probability'],
            c=self.customer_data['churn_probability'], 
            cmap='RdYlBu_r', 
            alpha=0.6
        )
        ax5.set_xlabel('Total Spent (£)')
        ax5.set_ylabel('Churn Probability')
        ax5.set_title('💎 Customer Value vs Churn Risk', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax5)
        
        # 6. Recency Distribution (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(self.customer_data['recency_days'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(self.customer_data['recency_days'].mean(), color='red', linestyle='--', 
                   label=f'Avg: {self.customer_data["recency_days"].mean():.0f} days')
        ax6.set_xlabel('Days Since Last Purchase')
        ax6.set_ylabel('Number of Customers')
        ax6.set_title('📅 Customer Recency Distribution', fontsize=14, fontweight='bold')
        ax6.legend()
        
        # 7. Top Products by Popularity (Bottom Left)
        ax7 = fig.add_subplot(gs[2, 0])
        top_products = self.product_info.head(10)
        bars = ax7.barh(range(len(top_products)), top_products['total_sold'])
        ax7.set_yticks(range(len(top_products)))
        ax7.set_yticklabels([desc[:25] + '...' if len(desc) > 25 else desc 
                            for desc in top_products['Description']], fontsize=9)
        ax7.set_xlabel('Total Sold')
        ax7.set_title('🛒 Most Popular Products', fontsize=14, fontweight='bold')
        ax7.invert_yaxis()
        
        # 8. Monthly Customer Acquisition (Bottom Middle)
        ax8 = fig.add_subplot(gs[2, 1])
        self.customer_data['first_purchase_month'] = pd.to_datetime(self.customer_data['first_purchase_date']).dt.to_period('M')
        monthly_acquisitions = self.customer_data['first_purchase_month'].value_counts().sort_index()
        ax8.plot(monthly_acquisitions.index.astype(str), monthly_acquisitions.values, marker='o', linewidth=2)
        ax8.set_xlabel('Month')
        ax8.set_ylabel('New Customers')
        ax8.set_title('📈 Monthly Customer Acquisition', fontsize=14, fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
        
        # 9. Business Impact Summary (Bottom Right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Calculate key metrics
        high_risk = self.customer_data[self.customer_data['risk_segment'] == 'High Risk']
        total_revenue_at_risk = high_risk['total_spent'].sum()
        avg_churn_prob = self.customer_data['churn_probability'].mean()
        
        summary_text = f"""
        🎯 BUSINESS IMPACT SUMMARY
        
        💰 Total Revenue: £{self.customer_data['total_spent'].sum():,.0f}
        
        🚨 High Risk Customers: {len(high_risk):,}
        
        💸 Revenue at Risk: £{total_revenue_at_risk:,.0f}
        
        📊 Average Churn Risk: {avg_churn_prob:.1%}
        
        🎯 Top Priority: Save {len(high_risk):,} customers
        worth £{total_revenue_at_risk:,.0f}
        
        🛒 Recommendation Engine: Ready for
        personalized retention campaigns
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('🤖 ULTIMATE CUSTOMER INTELLIGENCE DASHBOARD\nChurn Prediction + Product Recommendations', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        plt.savefig('ultimate_customer_intelligence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_action_plan(self):
        """Generate specific action plan for saving customers"""
        
        print("\n🎯 GENERATING CUSTOMER RETENTION ACTION PLAN...")
        print("=" * 60)
        
        # Focus on high-risk customers
        high_risk = self.customer_data[self.customer_data['risk_segment'] == 'High Risk'].copy()
        high_risk = high_risk.sort_values('total_spent', ascending=False)
        
        print(f"🚨 IMMEDIATE ACTION REQUIRED:")
        print(f"   💰 Revenue at Risk: £{high_risk['total_spent'].sum():,.0f}")
        print(f"   👥 High-Risk Customers: {len(high_risk):,}")
        print(f"   📅 Average Days Inactive: {high_risk['recency_days'].mean():.0f}")
        
        print(f"\n👑 TOP 10 HIGH-VALUE CUSTOMERS TO SAVE:")
        for i, (_, customer) in enumerate(high_risk.head(10).iterrows(), 1):
            # Get recommendations for this customer
            recs = self.get_recommendations_for_customer(customer['CustomerID'], 3)
            rec_text = ""
            if recs:
                rec_text = f" | Recommend: {recs[0]['description'][:30]}..."
            
            print(f"   {i:2d}. Customer {customer['CustomerID']:5.0f}: "
                  f"£{customer['total_spent']:6.0f} value, "
                  f"{customer['churn_probability']:.1%} risk, "
                  f"{customer['recency_days']:3.0f} days inactive{rec_text}")
        
        # Generate recommendations for high-risk customers
        print(f"\n🛒 PERSONALIZED RETENTION RECOMMENDATIONS:")
        sample_customers = high_risk.head(5)
        
        for _, customer in sample_customers.iterrows():
            recs = self.get_recommendations_for_customer(customer['CustomerID'], 3)
            print(f"\n   👤 Customer {customer['CustomerID']:.0f} (£{customer['total_spent']:.0f} value):")
            
            if recs:
                for j, rec in enumerate(recs, 1):
                    print(f"      {j}. {rec['description'][:40]:<40} £{rec['avg_price']:6.2f}")
            else:
                print("      No specific recommendations available - offer general promotions")
        
        return high_risk

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 LAUNCHING ULTIMATE CUSTOMER INTELLIGENCE SYSTEM!")
    print("=" * 60)
    
    # Initialize the system
    system = UltimateCustomerIntelligence()
    
    # Phase 1: Churn Prediction
    print("\n" + "="*50)
    print("🤖 PHASE 1: CHURN PREDICTION")
    print("="*50)
    
    customer_data = system.create_customer_features()
    auc_score = system.train_churn_model()
    
    # Phase 2: Recommendation Engine
    print("\n" + "="*50)
    print("🛒 PHASE 2: RECOMMENDATION ENGINE")
    print("="*50)
    
    product_data = system.build_recommendation_engine()
    
    # Create ultimate dashboard
    print("\n" + "="*50)
    print("📊 PHASE 3: ULTIMATE DASHBOARD")
    print("="*50)
    
    system.create_ultimate_dashboard()
    
    # Generate action plan
    print("\n" + "="*50)
    print("🎯 PHASE 4: ACTION PLAN")
    print("="*50)
    
    high_risk_customers = system.generate_action_plan()
    
    print(f"\n🎉 ULTIMATE CUSTOMER INTELLIGENCE SYSTEM COMPLETE!")
    print(f"=" * 60)
    print(f"✅ Churn Model Accuracy: {auc_score:.3f} AUC")
    print(f"✅ Recommendation Engine: Active")
    print(f"✅ Customer Intelligence Dashboard: Created")
    print(f"✅ Action Plan: Generated")
    print(f"\n💡 You've built enterprise-level customer intelligence!")
    print(f"🚀 This is the kind of work that gets data scientists hired at £60-90K+!")