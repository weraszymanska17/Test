# 📊  Data Visualization for Business Optimization 

## 🧭 Project Overview
This project demonstrates how to **manage Power BI environment transitions** and **data source migrations** using Microsoft SQL Server and MySQL.  
It replicates a real-world scenario where a BI developer needs to:
- Move reports from a **test environment** to **production**, and  
- Migrate Power BI reports from **SQL Server** to **MySQL**  

The goal was to ensure **data consistency**, **KPI accuracy**, and **smooth transition** between environments — a common challenge in enterprise BI projects.

## 🧠 Business Problem

A retail company tracks product demand and availability across stores.  
Reports were first developed using **test data in SQL Server**.  
Once validated, the reports needed to be:
1. Shifted to the **production SQL Server** environment  
2. Later migrated to a **MySQL** database used by the client  

The Power BI report monitors key supply chain and financial KPIs, helping management identify:
- Supply shortages  
- Profit/loss trends  
- Demand fulfillment rates  

## 🗃️ Dataset Description
### 1️⃣ Demand & Availability Table (`DA_Table`)
| Column | Description |
|--------|--------------|
| **OrderDate** | Date when a product was ordered |
| **ProductID** | Unique product identifier |
| **Availability** | Units available in stock on that date |
| **Demand** | Number of units demanded on that date |

### 2️⃣ Products Table (`Products`)
| Column | Description |
|--------|--------------|
| **ProductID** | Unique product ID |
| **ProductName** | Product name |
| **UnitPrice** | Unit price of the product |

> 🧮 The dataset represents daily product-level data, used to calculate operational and financial KPIs.

## 📊 Key Performance Indicators (KPIs)

### 🧱 Page 1: Supply & Demand Metrics
- **Average Daily Demand**  
- **Average Daily Availability**  
- **Total Supply Shortage (Unfulfilled Demand)**  

### 💰 Page 2: Financial Metrics
- **Total Profit**  
- **Total Loss**  
- **Average Daily Loss**  

Each KPI was implemented using **DAX Measures** to ensure dynamic aggregation and filtering.


## ⚙️ End-to-End Workflow

### 1. Test Environment (SQL Server)
- Imported raw data into **Microsoft SQL Server (Test Environment)**.  
- Prepared relationships between demand, availability, and product data.  
- Connected **Power BI Desktop** to the SQL Server test environment.  
- Created DAX measures for KPIs such as average demand, availability, and total supply shortage.  
- Designed the initial two-page Power BI report:
  - **Page 1:** Supply & Demand Metrics  
  - **Page 2:** Financial Performance Indicators  
- Validated all calculations and ensured data integrity in the test phase.

### 2. Transition to Production Environment
- Loaded verified data into the **Production SQL Server** database.  
- Cleaned and validated data to ensure accuracy before deployment.  
- Updated Power BI connections using the **Advanced Editor** in Power Query.  
- Reconnected all visuals and measures to the new data source.  
- Conducted a full validation to confirm the results matched the test environment.  
- Published the finalized report to the **Power BI Service**.

### 3. Migration to MySQL Database
- Installed and configured **MySQL Workbench** and **MySQL Connector**.  
- Imported production data into the **MySQL** environment.  
- Rebuilt equivalent queries for the MySQL structure.  
- Switched Power BI data source from SQL Server to MySQL via **Power Query Advanced Editor**.  
- Performed side-by-side validation of KPIs to ensure full consistency across environments.  
- Published the migrated report to the **Power BI Service**.  


## 🧰 Tools & Technologies
- Microsoft SQL Server – Managing test & production environment
- MySQL Workbench - Database migration and validation
- Power BI Desktop & Service – Modeling, visualization, and reporting
- Power Query – Data transformation, source switching
- DAX – Custom KPIs and advanced calculations


## 📈 Key Insights
- Demonstrated the ability to manage **multi-environment Power BI projects** from test to production.  
- Ensured **data consistency and KPI accuracy** throughout all transitions.  
- Strengthened skills in **data source reconfiguration** and **Power Query Advanced Editor** usage.  
- Improved understanding of **database interoperability** between SQL Server and MySQL.  
- Developed a workflow for **report validation, migration, and publishing** in Power BI Service.  
- Gained hands-on experience with **real-world BI deployment processes** and cross-environment data management.


## 🌐 Links
- [Dashboard in PDF](Project_2.pdf)
- [Power BI Dashboard](https://app.powerbi.com/groups/ae732784-0406-4839-9c1a-079d5bcd2d66/reports/b86f7967-08f2-41c4-ab4c-99adb63ba0a0/45d1c450d3eb65186867?experience=power-bi)
- [SQL Script - Test environment](sql_test.txt)
- [SQL Script - Production environment](sql_prod.txt)
- [MySQL Script](mysql_prod.txt)

## 🏷️ Tag: `Power BI` `SQL` `DataMigration` `MySQL` `Data Visualization`
