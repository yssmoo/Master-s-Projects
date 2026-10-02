# Start-ups & French Higher Education and Research (ESR)

**M1 ESA (Applied Econometrics & Statistics) — Advanced Python Programming, Université d'Orléans**

Authors: Abdellah Nait Akli, Yassir Motia
Supervisor: Mrs. Brosse

---

## Overview

This project analyses the French open dataset of **start-ups linked to higher education and research (ESR) institutions** — universities, research organisations such as CNRS, engineering schools, public research incubators, etc.

The goal is to understand and visualise the partnerships between public research institutions and the start-ups they work with, along four axes:

1. **Partnership organisation** — which start-ups are linked to a given institution, and which other institutions they co-partner with
2. **Geographical location** — how start-ups are distributed across regions, departments and municipalities
3. **Closed start-ups** — how many have ceased activity, and their average lifetime (overall and by region)
4. **Assets** — the types of assets (patents, software, etc.) produced by the start-ups and their relative frequency

The script ends with an **interactive terminal menu** and a **choropleth map extension** built with GeoPandas.

## Data

- **Source:** [data.gouv.fr — Les start-ups liées à l'enseignement supérieur et à la recherche académique française](https://www.data.gouv.fr/datasets/les-start-ups-liees-a-lenseignement-superieur-et-a-la-recherche-academique-francaise)
- **File expected:** `fr-esr-startup-esr-france.csv` (`;`-separated), placed in the same folder as the script
- **Region boundaries (map extension):** [france-geojson](https://france-geojson.gregoiredavid.fr/), loaded at runtime from its URL

Column names and category values (`Structures ESR liées`, `Région`, `Liste d'actifs`, `inactive`, …) are kept in French in the code, since they come directly from the source dataset.

## What the script does

### 1. Data preparation
- Parses date columns (`Date création`, `Date fermeture`, …) with `errors="coerce"` so invalid dates become `NaT`
- Converts low-cardinality text columns (`Statut`, `Région`, `Département`, `Commune`, `Liste d'actifs`) to `category`
- Handles **multi-valued columns** (comma-separated values) through `value_set()`, which returns the set of all distinct values in a column
- Saves the cleaned dataset to `df_startup_clean.csv`

### 2. Analysis functions

| Function | Purpose |
|---|---|
| `value_set(df, column)` | Set of all distinct values in a comma-separated column |
| `linked_startups(df, structure)` | Start-ups linked to an ESR structure (case-insensitive substring match) |
| `partner_structures(df, structure)` | Other ESR structures co-partnering with that structure's start-ups |
| `count_by_location(df, column)` | Number of unique companies (SIREN) per region / department / municipality |
| `partner_locations(df, structure, column)` | Same count, restricted to a structure's partner start-ups |
| `closed_startups(df)` | Inactive start-ups with a `lifetime` column (days), negative durations removed |
| `closed_partners(df, structure)` | Closed start-ups linked to a given structure, with their lifetime |
| `with_asset(df, asset_type)` | Companies owning at least one asset of a given type |
| `count_by_category(df, column, categories)` | Occurrences of each category in a column |
| `count_startups_with_asset(df, asset_type)` | Number of companies owning a given asset type |
| `plot_asset_distribution(df)` | Bar chart of asset-type shares among companies with at least one asset |
| `structure_report(df, name)` | Full text + chart report for one ESR structure |
| `map_by_region(df, title)` | Choropleth map of start-ups per metropolitan region (GeoPandas + contextily basemap) |

### 3. Interactive menu

```
  1. List ESR structures
  2. Full report for a structure
  3. Top regions / departments / municipalities
  4. Overall asset distribution (chart)
  5. Average lifetime of closed start-ups
  6. Map: all start-ups by region
  7. Map: start-ups linked to an ESR structure
  8. Quit
```

## Requirements

Python 3.9+ and:

```bash
pip install pandas matplotlib geopandas contextily
```

The map options (6 and 7) need an internet connection to download the region boundaries and basemap tiles.

## Usage

1. Download the CSV from the data.gouv.fr link above and save it as `fr-esr-startup-esr-france.csv` next to the script.
2. Run:

```bash
python RUN_ALL.py
```

The script first prints the exploratory results (data types, missing values, CNRS example, regional counts, lifetimes, asset shares) and shows two bar charts, then opens the interactive menu.

## Project structure

```
.
├── RUN_ALL.py                     # Full analysis + interactive menu
├── fr-esr-startup-esr-france.csv  # Input data (download separately)
├── df_startup_clean.csv           # Generated: cleaned dataset
└── README.md
```

## Notes and design choices

- **Matching on structure names** uses a case-insensitive substring search, so typing part of a name (e.g. `Sorbonne`) works, but it can also match several structures whose names share that substring.
- **Lifetime** is computed in days as `Date fermeture − Date création`. Records with a negative lifetime (closing date before creation date) are treated as data errors and dropped in `closed_startups()`.
- **Location counts** de-duplicate companies on `SIREN`, so a start-up appearing on several rows is counted once.
- **Overseas regions** are excluded from the map so that metropolitan France is readable.
- The assignment brief used French function names (`ensemble_valeurs`, `startups_liees`, `nb_par_lieu`, …). They were renamed to English for this repository; the mapping follows the order of the table above.
