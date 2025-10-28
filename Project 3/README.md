# 📊 Real Estate Market Analysis

## 🧭 Project Overview
This project explores Denmark’s housing market using **Google BigQuery** as the data warehouse and **Power BI** as the reporting and visualization tool.The dataset contains real estate transactions from Denmark, including property details, market metrics, and economic indicators such as interest and inflation rates.

The main goal was to:
- Connect Power BI to Google BigQuery for cloud-based data analysis  
- Perform **data understanding, cleaning, and transformation** using **Power Query Editor**  
- Build **interactive KPI dashboards** showing housing trends, sales performance, and price dynamics  



## 🧠 Business Problem

The Danish housing market involves multiple variables — price, interest rates, inflation, and regional dynamics.  
Real estate analysts and investors often struggle to identify:
- How offer prices compare to actual purchase prices,
- Which regions have the strongest year-over-year (YOY) growth,
- How interest and inflation affect housing demand.

This project aims to deliver **a clear analytical view of these relationships** to help stakeholders make informed business decisions.


## 🗃️ Dataset Description
The dataset was sourced from **Google BigQuery** and contains property-level data, including:
| Column | Description |
|--------|--------------|
| `date` | Date of the property transaction |
| `quarter` | Fiscal quarter of sale |
| `house_id` | Unique house identifier |
| `house_type` | Category of house (villa, apartment, farm, etc.) |
| `sales_type` | The type of sale, such as "new" or "resale" |
| `year_built` | The year the house was built |
| `purchase_price` | The price at which the house was purchased |
| `%_change_between_offer_and_purchase` | % difference between offer and purchase price |
| `no_rooms` | Number of rooms |
| `sqm` | Total area in square meters |
| `sqm_price` | Price per square meter |
| `address`, `region`, `city`, `area`, `zip_code` | Location details |
| `nom_interest_rate%` | Loan interest rate |
| `dk_ann_infl_rate%` | Yearly inflation rate in Denmark |
| `yield_mortgage_credit_bonds%` | Mortgage credit bond yield |


## ⚙️ End-to-End Workflow

### 1. Data Connection 
   Connected Power BI to Google BigQuery and imported the raw housing dataset.

### 2. Data Understanding & SQL Exploration  
   Used SQL queries in BigQuery to inspect and profile data for completeness and distribution.

### 3. Data Cleaning & Transformation (Power Query)  
   - Replaced missing values (e.g., inflation rate 1.85%, mortgage yield 1.47%).  
   - Adjusted data types and ensured consistency.  
   - Filtered invalid or null records.  
   - Enriched data with calculated fields.

### 4. Data Modeling & DAX Measures
   - YOY Sales Growth (`CALCULATE`, `YEAR`, `MAX`, `IF`, `BLANK`)
   - Median Sales Price Change by Region (`MEDIANX`)
   - Units Sold & 12-Month Sales (`CALCULATE`, `DATESINPERIOD`, `DISTINCTCOUNT`)
   - TotalYTD Sales & SQM Ratio (`TOTALYTD`, `SUM`, `ALLEXCEPT`)

### 5. Dashboard Development
   Designed three Power BI report pages:
   - **House Market Overview** – KPIs, Offer vs Purchase comparison, YOY trends  
   - **Sales Performance** – Regional sales, Key Influencers, Offer to SQM Ratio, Average Price SQM by region  
   - **House Type Analysis** – Offer/Purchase Price, Economic indicators, SQM distribution by house type  

### 6. Publishing & Sharing
   Published the report to Power BI Service and configured workspace for automatic refresh.



## 🧰 Tools & Technologies
- Google BigQuery – SQL data exploration
- Power BI Desktop & Service – Modeling, visualization, and reporting
- Power Query – Data cleaning and transformations 
- DAX – Custom KPIs and advanced calculations


## 📈 Key Insights   

The Danish housing market demonstrates a strong balance between **stability and regional diversity**. Throughout the analysis, three main perspectives emerged — *market behavior, regional performance, and property type dynamics* — each offering unique insights for both investors and policymakers.

---

### 🏠 Market Overview
Over the past year, the real estate market has shown **steady momentum**, with **77 units sold** and a total transaction value of **13 billion DKK**.  
Offer and purchase prices remain **closely aligned (±5%)**, indicating limited negotiation margins and a **seller-oriented market environment**.

While the overall market remains stable, transaction behavior varies by sales channel:

- **Auction sales** experienced a **+29% YoY growth**, highlighting a rising investor interest in faster, competitive sales methods.  
- **Regular and other sales types** declined moderately (−21%).  
- **Family-based transactions** dropped sharply (−75%), likely driven by demographic shifts and stricter mortgage requirements.

---

### 🌍 Regional Insights
The analysis reveals a **strong geographic concentration of value**:

| Region         | Total Sales | Share of Market | Avg. Price per m² |
|----------------|--------------|------------------|-------------------|
| Zealand        | 95bn DKK     | 35.6%            | 20.8K DKK/m²      |
| Jutland        | 81bn DKK     | 23.3%            | 13.5K DKK/m²      |
| Fyn & Islands  | 15bn DKK     | 8.0%             | 13.6K DKK/m²      |
| Bornholm       | 1bn DKK      | 1.8%             | 10.6K DKK/m²      |

These differences highlight **urban demand concentration** and **regional affordability gaps** — key indicators for investors and policy designers targeting housing balance and accessibility.

---

### 🏘️ Property Type Analysis
The *House Type Analysis* dashboard shows how **property type directly impacts investment potential**:

- **Farms and villas** deliver **high yields (4.2–4.6%)** and large areas (150–200 m²), ideal for long-term investors.  
- **Apartments** achieve the **highest price per m² (~28.7K DKK)**, reflecting **urban and rental demand strength**.  
- **Summerhouses** form a **low-cost niche**, driven by seasonal interest.

Across categories, minimal gaps between offer and purchase prices suggest a **transparent, efficient market with low speculation**.

---

### 💹 Financial Landscape
The broader economic context supports housing stability:

- Average **interest rate** ≈ 2%  
- Average **inflation rate** ≈ 1.9%  
- Typical **mortgage yield** ≈ 4–4.6%

This balance between financing costs and yield growth indicates a **mature, well-regulated housing ecosystem**.

---


## 🌐 Links
- [Dashboard in PDF]
- [Power BI Dashboard](https://app.powerbi.com/groups/me/reports/83f3aa54-997e-432d-95af-be68e6f52f22/7a07bb9a6364c4ae0800?experience=power-bi&clientSideAuth=0)

## 🏷️ Tag: `Power BI` `SQL` `RealEstateAnalytics` `BigQuery` `Data Visualization`
