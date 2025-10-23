# 📊  Loan Default Risk Analysis 

## 🧭 Project Overview

The purpose of this project was to develop a complete **Business Intelligence solution** to help a financial institution **identify high-risk borrowers** and **improve loan approval decisions**.  

The dataset contains information about loan applicants, their demographic details, financial status, loan characteristics, and repayment behavior.  

By analyzing key patterns in defaults, income levels, credit scores, and employment types, the goal is to support **data-driven decision-making** for future loan disbursement.

## 🧠 Business Problem

The bank faced a **high number of loan defaults**. Loan officers need **clear, fast insights** into which borrower segments are more likely to default.  

The BI solution answers:
- Which borrower profiles are more likely to default?  
- Which loan types or terms carry the highest risk?  
- How do factors like income, credit score, employment, and DTI ratio influence default rates?  
- How have loan trends and default rates evolved over time?

The dashboard enables bank officials to **quickly assess loan risk** and **optimize approval strategies**.

## 🗃️ Dataset Description

Each record represents a borrower who applied for a loan.  
Key fields include:

| Column | Description |
|--------|-------------|
| `Loan_ID` | Unique identifier for each loan |
| `Age` | Borrower’s age at loan issuance |
| `Income` | Borrower’s annual income |
| `Loan_Amount` | Amount of loan requested/approved |
| `Credit_Score` | Borrower’s creditworthiness score (300–850) |
| `Months_Employed` | Months with current employer |
| `Credit_Lines` | Number of active credit lines |
| `Interest_Rate` | Annual interest rate on the loan |
| `Loan_Term` | Duration of loan repayment (months) |
| `DTI_Ratio` | Debt-to-income ratio |
| `Education` | Highest education level |
| `Employment_Type` | Employment status (e.g., full-time, part-time, self-employed) |
| `Marital_Status` | Marital status |
| `Has_Mortgage` | Indicator of existing mortgage |
| `Has_Dependents` | Indicator of dependents |
| `Loan_Purpose` | Reason for taking out the loan |
| `Has_Cosigner` | Indicator if loan has a cosigner |
| `Default` | Whether borrower defaulted (0 = no, 1 = yes) |
| `Loan_Date` | Date of loan issuance |


## ⚙️ End-to-End Workflow

### 1. Environment Setup
- Imported raw CSV dataset into SQL Server.  
- Created a **Power BI Dataflow** in Power BI Service for centralized ETL.

### 2. Data Preparation & Profiling
- Used **Power Query** to clean and profile data.  
- Handled data types, renamed columns, standardized categories.  
- Performed data validation checks to ensure model accuracy.

### 3. Data Modeling & DAX Calculations
- **Loan Amount by Purpose** — `SUMX`, `FILTER`, `NOT`, `ISBLANK`  
- **Average Income by Employment Type** — `CALCULATE`, `AVERAGE`, `ALLEXCEPT`  
- **Default Rate by Segment** — `COUNTROWS`, `DIVIDE`, `FILTER`  
- **Average Loan by Age Group** — `AVERAGEX`, `VALUES`  
- **Default Rate by Year** — `CALCULATE`, `ALLEXCEPT`  
- **Median Loan Amount by Credit Score Category** — `MEDIANX`  
- **YOY and YTD Loan Amount & Default Rate** — `DATEADD`, `CALCULATE`  
- **Decomposition Tree Analysis** — `SWITCH`

### 4. Data Visualization & Storytelling
- **KPIs**: Total Loan Amount, Average Income, Default Rate, Number of Borrowers  
- **Charts**:
  - Donut Chart — Average loan by age group & marital status  
  - Clustered Column — Loan amount vs. dependents, mortgage  
  - Line Charts — Trends in loan amount and default rate
  - Ribbon Chart - YTD Loan Amount by Credit Score Bins and Marital Status 
  - Decomposition Tree — Key drivers of default & loan value  

### 5. Automation & Deployment
- Configured **Scheduled Refresh** in Power BI Service.  
- Implemented **Incremental Refresh** for better performance.  
- Published dashboard for business users.

## 🧰 Tools & Technologies

- **Microsoft SQL Server** – Data storage & querying  
- **Power BI Desktop & Service** – Modeling, visualization, and reporting  
- **Power Query** – Data preparation  
- **DAX** – Advanced calculations and measures    
- **Incremental Refresh** – Performance optimization

## 📈 Key Insights
- Employment Type Matters: Full-time employees have the lowest default rate (2.36%), while unemployed borrowers have the highest (3.39%).

- Credit Score Risk: Borrowers with low or medium credit scores take larger loan amounts.

- Demographics: Adults and middle-aged borrowers represent the largest loan amounts. Borrowers with Bachelor’s and High School education levels represent the largest number of loans. Marital status does not significantly affect the average loan amount.

- Loan Purpose: The largest volumes are for home and business loans, while education and auto loans represent smaller shares.

- Time Trends: YOY Loan Amount increased significantly in 2015 and 2018, suggesting periods of higher credit demand.

- Key Drivers: Income, employment type, credit score, and DTI ratio are the most significant factors behind defaults.


## 🌐 Links
- [See Dashboard in PDF](Project_1/Loan_Default_Risk_Analysis.pdf)
- [Power BI Dashboard](https://app.powerbi.com/groups/d9d52c35-73cc-477f-b0d9-e70a653430cd/reports/93f81295-d66d-4f49-be43-a9452e295244/7054507e0b49540e2a84?experience=power-bi)

## 🏷️ Tag: `Power BI` `SQL` `Loan Analysis` `Risk Analytics` `Data Visualization`
