# 📊 Men's Fashion Brand Performance Analysis (Insight BI)

## 🧭 Project Overview
This project focuses on analyzing sales, discount, and profitability performance across multiple men’s fashion brands.  
Using **Azure SQL Database** for data storage, **SQL** for cleaning, and **Power BI** for visualization, the goal was to identify which brands are driving the most value through discounts, sales volume, and profit margins.

The dashboard — **Insight BI: Men’s Collection** — provides a consolidated, interactive view of top and bottom-performing brands, enabling data-driven decision-making for merchandising and pricing strategies.

## 🧠 Business Problem

The retail fashion industry operates in a competitive market where understanding brand performance is essential for optimizing pricing and profitability.  
The business wanted to answer:

- Which brands offer the highest discounts and does it impact profit margins?  
- Which brands generate the highest average sales prices?  
- Which brands have the greatest product variety, and how does that relate to performance?  
- What are the least profitable brands, and how can pricing strategy be improved?



## 🗃️ Dataset Description
The dataset contains sales transaction details for men’s fashion products, including:

| Column | Description |
|--------|--------------|
| `Brand` | Brand name |
| `Title` | Product desctiption |
| `Original Price` | Initial listed price before discount |
| `Sales Price` | Final selling price |

Before analysis, data cleaning was performed to remove invalid symbols (e.g., “?”) and ensure numeric consistency in price columns.


## ⚙️ End-to-End Workflow

### 1. Data Ingestion
   - Created a **Free Azure SQL Database**.
   - Imported raw product and sales data into Azure.
### 2. Data Cleaning (SQL)
   - Replaced invalid characters (`?`) with blanks in price columns.
   - Trimmed whitespace and standardized text.
### 3. Data Connection
   - Connected Power BI directly to Azure SQL Database using both “Database” and “Microsoft Account” options.
### 4. Data Transformation (Power BI)
   - Created new DAX measures:
     - `Discount %`
     - `Profit %`
     - `Cost Price`
### 5. Visualization
   - Built multi-page dashboard including:
     - **Top 5 Brands by Discount %, Profit %, Sales Price**
     - **Top 5 Brands by Product Variety**
     - **Bottom 5 Brands by Profit %**
   - Used bar, donut, ribbon, and area charts for comparative analysis.
### 6. Deployment
   - Published and shared the dashboard through **Power BI Service** as an app.



## 🧰 Tools & Technologies
- Microsoft Azure SQL Database – Cloud data storage & query execution
- Azure Portal - Cloud configuration & access control 
- Power BI Desktop & Service – Modeling, visualization, and reporting
- DAX – Custom KPIs and advanced calculations
- SQL - Data cleaning, transformation, and validation


## 📈 Key Insights   
The analysis of men’s fashion brands revealed distinct pricing and profitability patterns that shed light on how different strategies impact overall performance.

### 1️⃣ Discounts and Their Impact on Profit Margins
**The Indian Garage Co.** and **British Club** stand out as the most aggressive discounters, with average discounts exceeding **65%**.  While these strategies drive sales volume and customer engagement, they also compress profit margins. Despite strong discount-driven visibility, neither brand ranks among the top performers in profitability — indicating a trade-off between short-term sales boosts and long-term margin stability.

---
### 2️⃣ High-Value Brands by Sales Price
When focusing on **average sales price**, luxury and semi-premium labels dominate the leaderboard:  **ARMANI EXCHANGE**, **BROOKS BROTHERS**, **Terra Luna**, and **Scotch & Soda** demonstrate a clear premium strategy. 
These brands manage to sustain strong revenue per unit even with limited discounting — suggesting efficient brand positioning, perceived exclusivity, and effective targeting of high-income consumers.

---
### 3️⃣ Product Variety and Market Reach
Brands like **The Indian Garage Co.** and **U.S. Polo Assn.** lead in **product variety**, offering a broad assortment across multiple fashion categories. A wide catalog correlates with stronger brand recognition and market share but may also lead to operational complexities and diluted profit margins.  Their strategy focuses on reach and volume — effective for visibility but potentially costly without optimization of discount depth and stock rotation.

---
### 4️⃣ Least Profitable Brands and Strategic Recommendations
At the lower end of profitability, **The Souled Store**, **ADWYN PETER**, and **Be Active X AG** report minimal or negative profit margins.  These brands likely over-index on discounting or operate in highly competitive price bands with limited brand differentiation.

**Recommended Strategy:**
- Implement **data-driven pricing tiers** to balance discounts and margins.  
- Evaluate **product mix** to focus on best-performing SKUs rather than expanding unprofitable lines.  
- Introduce **targeted promotions** instead of blanket discounts to sustain perceived brand value.


## 🌐 Links
- [Dashboard in PDF](Men's_Fashion_Brand_Analysis.pdf)
- [Power BI Dashboard](https://app.powerbi.com/groups/7e377593-ac2e-4c7e-b56f-cddaf434f21e/reports/1b604c60-4dd5-4e0b-bbc4-b81be65ac6cc/e8234cb802b7b1168003?experience=power-bi&clientSideAuth=0)

## 🏷️ Tag: `Power BI` `SQL` `BrandsAnalysis` `AzureSQL` `Data Visualization` `CloudAnalytics`
