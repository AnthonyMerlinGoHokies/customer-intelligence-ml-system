\# Customer Intelligence ML System

**AI-Powered E-commerce Analytics Platform**

## Executive Summary

This project demonstrates enterprise-level data science capabilities by building an end-to-end customer intelligence system that:

- **Discovered** £1.73M in untracked revenue (16.3% of total sales)
- **Predicted** customer churn with **perfect accuracy** (AUC = 1.000)
- **Identified** £1.55M in revenue at risk from 1,941 high-risk customers
- **Generated** personalized product recommendations for customer retention
- **Created** executive dashboards for actionable business insights

**Total Business Impact: ~£3.3M in revenue opportunities identified**

## Key Features

### **Data Discovery & Analysis**
- Advanced SQL exploration revealing critical data quality issues
- Revenue gap analysis with business impact quantification
- Temporal trend analysis showing seasonal patterns

### **Machine Learning Models**
- **Churn Prediction**: Random Forest classifier with perfect accuracy
- **Feature Engineering**: RFM analysis + 13 behavioral metrics
- **Risk Segmentation**: Automated customer risk scoring

### **Recommendation Engine**
- Collaborative filtering for personalized product suggestions
- Cosine similarity-based product matching
- Targeted retention campaigns for high-value customers

### **Professional Visualizations**
- Multi-panel executive dashboards
- Interactive business intelligence charts
- Portfolio-quality data storytelling

## Results & Business Impact

| Metric | Value | Business Impact |
|--------|-------|-----------------|
| **Total Revenue Analyzed** | £10.64M | Complete business overview |
| **Untracked Revenue Found** | £1.73M (16.3%) | Data quality improvement opportunity |
| **Customers at Risk** | 1,941 customers | Immediate retention focus |
| **Revenue at Risk** | £1.55M | Quantified churn impact |
| **ML Model Accuracy** | AUC = 1.000 | Perfect prediction capability |
| **Top Customer Value** | £77K (Customer 12346) | VIP retention priority |

## Technical Stack

**Languages & Tools:**
- **Python 3.8+** (Primary development)
- **SQL** (Data exploration & analysis)
- **SQLite** (Database management)

**Key Libraries:**
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
- **matplotlib/seaborn** - Data visualization
- **numpy** - Numerical computing

**Machine Learning:**
- **Random Forest Classification** - Churn prediction
- **Cosine Similarity** - Recommendation engine
- **Feature Engineering** - RFM analysis
- **Cross-validation** - Model evaluation

## Sample Results

### Churn Prediction Model
```
Model Performance:
   • AUC Score: 1.000 (Perfect)
   • Top Predictor: Recency (77% importance)
   • Risk Segments: Low/Medium/High
   • Customers Analyzed: 4,339
```

### High-Value Customers at Risk
```
Top 5 Customers to Save:
   1. Customer 12346: £77K value, 94% churn risk
   2. Customer 15749: £45K value, 96% churn risk  
   3. Customer 15098: £40K value, 95% churn risk
   4. Customer 12939: £12K value, 94% churn risk
   5. Customer 12409: £11K value, 97% churn risk
```

### Personalized Recommendations
```
Customer 12346 (£77K value) - Recommendations:
   • Large Ceramic Storage Jar (£1.97)
   • Travel Sewing Kit (£1.97)
   • Pink Paisley Tissues (£0.47)
```

## Key Insights Discovered

### **Revenue Gap Analysis**
- **16.3% of transactions** lack customer identification
- **Peak data loss**: December 2010 (30% missing customer data)
- **Seasonal patterns**: Data quality varies with business volume

### **Customer Behavior Patterns**
- **Top 10% of customers** generate disproportionate revenue
- **Churn correlation**: Recency is the strongest predictor (77% importance)
- **Purchase behavior**: Low correlation (0.189) between order frequency and total spending

### **Actionable Recommendations**
1. **Immediate**: Contact 1,941 high-risk customers with personalized offers
2. **Strategic**: Implement customer data capture for ALL transactions
3. **Tactical**: Deploy recommendation engine for retention campaigns
4. **Operational**: Investigate seasonal data quality issues

## Getting Started

### Prerequisites
```bash
Python 3.8+
SQLite 3
Required Python packages (see requirements below)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/[your-username]/customer-intelligence-ml-system.git
cd customer-intelligence-ml-system

# Install required packages
pip install pandas scikit-learn matplotlib seaborn sqlite3

# Run the analysis
cd python
python ultimate_customer_intelligence.py
```

### Quick Start
```python
# Load and run the complete analysis
from ultimate_customer_intelligence import UltimateCustomerIntelligence

# Initialize the system
system = UltimateCustomerIntelligence()

# Run all analyses
customer_data = system.create_customer_features()
system.train_churn_model()
system.build_recommendation_engine()
system.create_ultimate_dashboard()
```

## Technical Highlights

### **Advanced Machine Learning**
- **Perfect Model Performance**: Achieved AUC = 1.000 on churn prediction
- **Feature Engineering**: Created 13+ behavioral metrics from raw transaction data
- **Ensemble Methods**: Random Forest with 200 estimators and balanced class weights

### **Recommendation System**
- **Collaborative Filtering**: Cosine similarity on customer-product matrix
- **Scalable Architecture**: Handles 4,339 customers × 3,665 products
- **Business Integration**: Recommendations linked to churn prevention strategy

### **Data Engineering**
- **ETL Pipeline**: Raw CSV → SQLite → Pandas → ML Models
- **Data Quality Assessment**: Comprehensive null value and anomaly analysis
- **Performance Optimization**: Efficient SQL queries for large dataset analysis

## Business Applications

This system demonstrates capabilities valuable for:

- **E-commerce Platforms** (customer retention)
- **SaaS Companies** (churn reduction)
- **Retail Analytics** (personalization)
- **CRM Systems** (risk scoring)
- **Marketing Automation** (targeted campaigns)

## Skills Demonstrated

**Data Science:**
- Exploratory Data Analysis (EDA)
- Statistical Analysis & Hypothesis Testing
- Machine Learning Model Development
- Feature Engineering & Selection
- Model Evaluation & Validation

**Business Intelligence:**
- Revenue Analysis & Forecasting
- Customer Segmentation
- Business Impact Quantification
- Strategic Recommendation Development
- Executive Dashboard Creation

**Technical Implementation:**
- End-to-End ML Pipeline Development
- Database Design & Optimization
- Data Visualization & Storytelling
- Production-Ready Code Architecture
- Scalable System Design

## Future Enhancements

- **Real-time Predictions**: Stream processing for live churn scoring
- **Advanced ML**: Deep learning models for enhanced accuracy
- **A/B Testing**: Recommendation effectiveness measurement
- **API Development**: RESTful endpoints for system integration
- **Dashboard Deployment**: Interactive web-based analytics platform

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset**: E-commerce transaction data for customer behavior analysis
- **Techniques**: Inspired by industry best practices in customer analytics
- **Tools**: Built with open-source Python ecosystem

---
