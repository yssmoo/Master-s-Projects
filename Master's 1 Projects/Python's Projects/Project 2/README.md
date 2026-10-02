# Open Food Facts — Nutri-Score & Additives Analysis

**M1 ESA (Applied Econometrics & Statistics) — Advanced Python Programming, Université d'Orléans**

Authors: Abdellah Nait Akli, Yassir Motia
Supervisor: Mrs. Brosse

---

## Overview

This project queries the [Open Food Facts](https://world.openfoodfacts.org/) **REST API** to analyse processed food products by category. The user picks a food type (from a preset list or free text). The program then downloads every matching product, cleans the data, and produces statistics and charts on:

- the **Nutri-Score** distribution (A → E)
- the **number of additives** per product and how it relates to the Nutri-Score
- average **sugar, fat, salt and energy** content, and the correlations between them
- **controversial additives** (extension, see below)
- a side-by-side **comparison of two food types**

## Data source

- **API:** Open Food Facts search endpoint v2: `https://world.openfoodfacts.org/api/v2/search` ([documentation](https://openfoodfacts.github.io/openfoodfacts-server/api/))
- **Fields retrieved:** `code` (barcode), `categories_tags_fr`, `nutriscore_grade`, `additives_tags`, `nutriments`
- **Pagination:** 100 products per page (API maximum), capped at 20 pages (2,000 products) per query, with a short pause between requests

Search terms such as `"Pâte à tartiner chocolat et noisettes"` or `"jus de fruits"` are kept in French, since they are passed to the API as-is and match the French category names in Open Food Facts.

## What the script does

### Part 1 — Data preparation (chocolate-hazelnut spread example)
- `get_spreads_page1()`: first page of results, keeping only barcode, category and Nutri-Score
- `get_spreads_full()`: loops over all pages to retrieve every matching product

### Part 2 — Data processing

| Function | Purpose |
|---|---|
| `clean_dataframe(df)` | Keeps only products with a valid Nutri-Score (a–e), counts additives, and extracts sugar, fat, salt and energy per 100 g |
| `get_food_dataframe(food_type)` | Downloads all products for a given food type, including additives and nutrients |
| `count_nutriscore(df)` | Number of products per Nutri-Score grade |
| `count_additives_per_product(df)` | Number of additives for each barcode |
| `global_statistics(df)` | Average sugar, fat, salt, additives and energy |
| `nutrition_correlation(df)` | Correlation matrix between sugar, fat, salt and number of additives |

### Part 3 — Presentation of results

| Function | Chart |
|---|---|
| `pie_nutriscore(df)` | Pie chart of the Nutri-Score distribution |
| `bar_nutriscore(df)` | Bar chart of products per Nutri-Score |
| `boxplot_additives(df)` | Boxplot of the number of additives by Nutri-Score |
| `histogram_sugar(df)` | Histogram of sugar content |
| `heatmap_correlation(df)` | Heatmap of the nutritional correlation matrix |

### Extension — Controversial additives
This part cross-references the product additives (`additives_tags`, e.g. `en:e250`) with a list of controversial E-numbers:

- `load_controversial_additives()`: tries to download a list from GitHub, and otherwise falls back to a built-in list of 9 additives (E211, E220, E249–E252 nitrites/nitrates, E407 carrageenan, E476 PGPR, E621 MSG)
- `count_controversial_additives(df)`: adds the number of controversial additives per product
- `analyse_controversial_additives(df)` / `plot_controversial_additives(df)`: average count by Nutri-Score, printed and plotted

`compare_products(p1, p2)` also compares the average sugar and additive counts of two food types.

## Requirements

Python 3.9+ and:

```bash
pip install requests numpy pandas matplotlib seaborn
```

An internet connection is required, since all data is fetched live from the API.

## Usage

```bash
python RUN_ALL.py
```

```
=== Open Food Facts Analysis ===

Available products:
  1 - pizza
  2 - cereales
  3 - biscuits
  4 - jus de fruits
  5 - pate a tartiner chocolat et noisettes
  6 - yaourt
  7 - chips
  8 - chocolate
  0 - Enter a product manually
```

After you choose a food type, the script downloads the products and prints the statistics. It then shows six charts one after another (close each window to see the next one). Finally, it offers a comparison with a second product.

A full run can take a minute or two for large categories, because of the pagination and the pauses between requests.

## Project structure

```
.
├── RUN_ALL.py   # Full pipeline: API calls, cleaning, analysis, charts, menu
└── README.md
```

## Notes and limitations

- **Controversial-additives source:** the GitHub URL used by the extension (`nicklockwood/FoodAdditives`) currently returns a 404 error. In practice the script therefore always uses the built-in fallback list of 9 additives. That list is a hand-picked selection of commonly debated additives, not an official regulatory classification. A sturdier extension would use an official dataset, such as the European Commission food additives database.
- **Sample size:** results are capped at 2,000 products per query (`MAX_PAGES = 20`). Increase `MAX_PAGES` for a complete extraction of large categories.
- **Search relevance:** the API's full-text search can return products that are only loosely related to the query. Results depend on the exact search term and its language.
- **Missing data:** products without a valid Nutri-Score are dropped. Missing nutrient values are left as `NaN` and ignored in the averages.
- The assignment brief used French names (e.g. `df_pates_a_tartiner`). Functions and variables were renamed to English for this repository.
