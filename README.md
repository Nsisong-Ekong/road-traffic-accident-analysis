# Road Traffic Accident Analysis and Predictive Modeling

**Author:** Nsisong Patrick Ekong  
**Module:** Big Data and Data Mining  
**Institution:** University of Hull  

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methods & Analysis Tasks](#methods--analysis-tasks)
- [Key Findings](#key-findings)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Dependencies](#dependencies)
- [References](#references)

---

## Overview

Road traffic accidents (RTAs) are a critical public health challenge, responsible for approximately 1.3 million deaths annually worldwide (WHO, 2024). This project presents a comprehensive data-driven analysis of road traffic accident data from the United Kingdom, with the dual aim of identifying the key factors that influence accident severity and building predictive models to forecast future accident trends.

The study combines classical statistical analysis with modern machine learning techniques — including association rule mining, time series forecasting, spatial analysis, and social network analysis — to uncover actionable patterns in a large-scale accident database. The findings are intended to support policymakers, traffic management authorities, and urban planners in designing safer, more adaptive road environments.

---

## Dataset

The dataset is a **SQLite database** (`accident_data_v1.0.0_2023.db`) containing four relational tables of UK road traffic accident records spanning **2017 to 2020**.

| Table | Columns | Rows | Description |
|---|---|---|---|
| **accident** | 36 | 461,352 | Core accident records: severity, road type, weather, light conditions, date and time |
| **casualty** | 19 | 600,332 | Casualty details: severity, type (driver/pedestrian/passenger), age, gender, area |
| **lsoa** | 7 | 34,378 | Geographic identifiers: LSOA codes, coordinates, region names |
| **vehicle** | 28 | 849,091 | Vehicle-specific data: vehicle type, driver age, manoeuvres at time of accident |

### Key Variables

- **accident_severity:** Fatal, Serious, or Slight
- **road_type:** Roundabout, One-way street, Dual carriageway, Single carriageway, Slip road
- **weather_conditions:** Fine, Rain, Snow, Fog/mist, High winds (various combinations)
- **light_conditions:** Daylight, Darkness (with/without lighting)
- **road_surface_conditions:** Dry, Wet, Icy, Flooded
- **Temporal fields:** Date, time of day, day of week
- **police_force:** Police area responsible for the region
- **lsoa_of_accident_location:** Geographic LSOA code for spatial analysis

> **Note:** The SQLite database is not included in this repository due to size constraints. It can be sourced from the [UK Department for Transport – Road Safety Data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data). Place the file as `accident_data_v1.0.0_2023.db` in the root project directory before running the notebook.

---

## Project Structure

```
road-traffic-accident-analysis/
│
├── Accident Data Report Codes.ipynb   # Main analysis notebook (all tasks)
├── accident_data_v1.0.0_2023.db       # SQLite database (not included — see Dataset section)
├── Accident Data Report.pdf           # Full academic report
├── requirements.txt                   # Python dependencies
└── README.md
```

All analysis is contained within a **single Jupyter notebook**, structured into clearly labelled sections corresponding to each analytical task.

---

## Methods & Analysis Tasks

The notebook is organised around the following ten analytical tasks:

**1. Database Connection & Table Overview**  
Connects to the SQLite database using `sqlite3`, retrieves all four tables, and performs initial schema inspection — listing column names, data types, and row counts.

**2. Data Description & Exploration**  
Loads data from each table into pandas DataFrames, checks for missing values, and produces summary statistics. Data for 2020 is loaded separately to support regional and LSOA-level analyses.

**3. Initial Data Exploration & Summary Analysis**  
Visualises accident severity distribution, accident counts by road type, and accident counts by weather condition. Bar charts are produced for each breakdown.

**4. Accident Trends by Time and Demographics**  
Analyses accident frequency by hour of day and day of week for all accidents, broken down further by motorcycle engine category (≤50cc, 50–125cc, 125–500cc, 500cc+) and pedestrians.

**5. Association Rule Mining — Apriori Algorithm**  
Applies the Apriori algorithm (via `mlxtend`) on a 10,000-record sample with one-hot encoding of categorical features. Generates and ranks 178 association rules by support, confidence, and lift. Includes confidence vs lift and support vs lift scatter plots.

**6. Spatial Analysis — Hull, Humberside & East Riding of Yorkshire**  
Filters accident data for the local region. Produces scatter maps coloured by severity and Kernel Density Estimation (KDE) heatmaps using `scipy.stats.gaussian_kde` to identify accident hotspots.

**7. Three Policing Areas — Weekly Accident Forecasting**  
Identifies the top three UK police forces by accident count (2017–2019): Metropolitan Police, West Midlands, and Kent. Fits a **SARIMAX(1,1,1)(1,1,1,52)** model to each and forecasts weekly accident counts for the next 52 weeks.

**8. Top Hull LSOAs — Daily Accident Forecasting**  
Identifies the three Hull LSOAs with the highest accident counts in Q1 2020 (E06000010, E06000011, E06000013). Confirms stationarity with the Augmented Dickey-Fuller (ADF) test, then fits an **ARIMA(1,0,1)** model to forecast daily accidents for July 2020 using January–June 2020 as training data.

**9. Social Network Analysis**  
Constructs a social network using `networkx` by merging accident, casualty, and vehicle tables on the accident index. Nodes represent casualties, vehicles, and locations; edges represent their shared involvement in accidents. Reports network density, node count, edge count, and average degree.

**10. Edge Centrality & Community Detection**  
Calculates edge betweenness centrality across the network and plots its distribution. Applies two community detection algorithms — **Louvain** (`python-louvain`) and **Girvan-Newman** (`networkx`) — and compares their cluster structures visually and numerically.

---

## Key Findings

### Accident Severity
- **79.83%** Slight · **18.78%** Serious · **1.39%** Fatal (out of 461,352 total accidents)

### Road Type Risk
- Dual carriageways linked to serious accidents with 87.16% confidence (Apriori rule)
- Single carriageways have the highest serious accident proportion among carriageway types (20.09%)
- Roundabouts are the safest road type — slight accidents account for 86.05%

### Weather Conditions
- Fine weather (no wind) accounts for 79.55% of accidents, but fog/mist conditions yield the highest **fatal rate at 3.80%**
- Raining with high winds produces 2.34% fatal accidents vs. 1.42% in fine conditions

### Temporal Patterns
- **Peak hour:** 17:00 with 40,307 accidents (evening rush hour)
- **Peak day:** Friday (75,211 accidents) · Lowest: Sunday (52,028)
- **Pedestrians:** Peak at 15:00 (9,003 incidents), aligning with school run hours
- **Motorcycles (50–500cc):** Peak on Fridays (commuter pattern)
- **Motorcycles (500cc+):** Peak on Sundays (recreational riding pattern)

### Forecasting
- **Top policing areas:** Metropolitan Police (98,092 accidents), West Midlands (20,519), Kent (16,789)
- SARIMAX forecasts show seasonal peaks in summer and dips during holiday periods
- **Top Hull LSOAs (Q1 2020):** E06000010 (603), E06000011 (495), E06000013 (306)
- ARIMA forecasts predict stable daily accident counts through July 2020 (~750, ~647, ~386 respectively)
- Accident counts in Hull dropped to 1,709 in 2020, down from ~2,300/year (attributed to COVID-19)

### Social Network
- Network: **286 nodes**, **844 edges**, density = **0.0207** (sparse)
- Average degree: **5.90** connections per node
- Edge betweenness centrality: 0.0001 – 0.0061 (a small number of edges are critical connectors)
- **Louvain:** 3 clusters — 133, 68, and 85 nodes
- **Girvan-Newman:** 2 clusters — 285 nodes and 1 node

---

## Installation & Setup

### Prerequisites

- Python 3.12.4 or higher (developed and tested on 3.12.4)
- Jupyter Notebook or JupyterLab

### Clone the Repository

```bash
git clone https://github.com/your-username/road-traffic-accident-analysis.git
cd road-traffic-accident-analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add the Database

Download the UK Road Safety Data from the [Department for Transport](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) and place the SQLite database in the **same folder as the notebook**:

```
road-traffic-accident-analysis/
├── Accident Data Report Codes.ipynb
└── accident_data_v1.0.0_2023.db    ← place here (same directory as notebook)
```

The notebook connects using a relative path (`sqlite3.connect('accident_data_v1.0.0_2023.db')`), so no path changes are needed — it will work on any operating system as long as the database is in the same folder.

---

## Usage

Launch the notebook and run all cells from top to bottom:

```bash
jupyter notebook "Accident Data Report Codes.ipynb"
```

The notebook is self-contained. Each section is clearly labelled with its corresponding analytical task. All database queries, visualisations, model fitting, and outputs are executed within the single notebook file.

---

## Results Summary

| Task | Method | Key Output |
|---|---|---|
| Severity distribution | Descriptive stats | 79.83% Slight, 18.78% Serious, 1.39% Fatal |
| Road type risk | Bar charts + Apriori | Dual carriageways → highest serious accident rate |
| Temporal trends | Trend analysis | Peak: Friday 17:00; Trough: Sunday early morning |
| Association rules | Apriori (mlxtend) | 178 rules; daylight + dry road → slight (dominant pattern) |
| Spatial hotspots | KDE / Scatter maps | Hull city centre confirmed as highest-density zone |
| Policing area forecast | SARIMAX(1,1,1)(1,1,1,52) | Metropolitan Police: 125–558 weekly accidents projected |
| LSOA forecast | ARIMA(1,0,1) | E06000010: ~750 daily accidents forecast for July 2020 |
| Network structure | NetworkX SNA | 286 nodes, 844 edges, density 0.0207 |
| Edge centrality | Betweenness centrality | Range: 0.0001 – 0.0061 |
| Community detection | Louvain + Girvan-Newman | Louvain: 3 clusters; Girvan-Newman: 2 clusters |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
mlxtend
statsmodels
networkx
python-louvain
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn mlxtend statsmodels networkx python-louvain jupyter
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

---

## References

1. Ahmed, S. K., et al. (2023). Road traffic accidental injuries and deaths: A neglected global health issue. *Health Science Reports*, 6(5), e1240.
2. World Health Organization. (2024). *Advocating for road safety*. Retrieved from https://www.who.int
3. Kumar, S. S., et al. (2024). Economic Burden of Road Traffic Accidents in India. *Health Technology Assessment Resource Centre, ICMR*.
4. Litman, T. (2024). *Evaluating transportation equity*. Victoria Transport Policy Institute.
5. Fernandes, R., Hatfield, J., & Job, R. F. S. (2010). Differential predictors of speeding and drink-driving. *Transportation Research Part F*, 13(3), 179–196.
6. Hammad, H., et al. (2019). Environmental factors affecting road traffic accident frequency. *Environmental Science and Pollution Research*.
7. United Nations. (2015). *Transforming our world: The 2030 Agenda for Sustainable Development*.

---

## License

This project was developed as part of an academic assignment at the University of Hull. The code is available for educational and research purposes. Please cite appropriately if you use or build upon this work.
