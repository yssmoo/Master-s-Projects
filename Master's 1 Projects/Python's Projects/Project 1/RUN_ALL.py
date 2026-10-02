# Project 1: Study of the links between start-ups and French higher education
#            and research (ESR) institutions
# Abdellah NAIT AKLI & Yassir MOTIA
# M1 ESA — Advanced Python Programming
# Professor: Mrs. Brosse
#
# Note: column names and category values (e.g. "Structures ESR liées", "Région",
# "inactive") are kept in French because they come directly from the
# data.gouv.fr dataset.

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx


# Path to the raw dataset (download it from data.gouv.fr, see README)
DATA_FILE = "fr-esr-startup-esr-france.csv"
CLEAN_FILE = "df_startup_clean.csv"

# Shortcut for the most frequently used column
ESR_COL = "Structures ESR liées"
ASSETS_COL = "Liste d'actifs"


############################ 2.1 – DATA IMPORT AND PREPARATION ##############################

### 2.1.1 Loading
data = pd.read_csv(DATA_FILE, delimiter=";")

### 2.1.2 Column typing
# Dates: errors="coerce" turns invalid dates into NaT instead of raising an error
date_cols = ["Date création", "Date fermeture", "Date observation état", "Date accompagnement IRP"]

for col in date_cols:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors="coerce", yearfirst=True)

## Categorical columns
cat_cols = ["Statut", "Région", "Département", "Commune", ASSETS_COL]
for col in cat_cols:
    if col in data.columns:
        data[col] = data[col].astype("category")

# Mixed columns (str + numeric) are left as object

### 2.1.3 Technical summary
print(data.dtypes)
print(data.isna().sum())


### 2.1.4 Extract the unique values of a multi-valued column
def value_set(df, column):
    """Return the set of all values found in the column after splitting on ','."""
    if column not in df.columns:
        return set()
    values = set()
    for row in df[column].dropna().astype(str):
        # split + strip to cleanly break "val1, val2, val3" apart
        values.update(v.strip() for v in row.split(",") if v.strip())
    return values


### 2.1.5 Collect the value sets of every multi-valued column
multi_valued_cols = [ESR_COL,
                     "Identifiant Structures ESR liées",
                     ASSETS_COL,
                     "Incubateur de la recherche publique"]

value_sets = {}
for col in multi_valued_cols:
    if col in data.columns:
        value_sets[col] = value_set(data, col)
        print(f"{col} → {len(value_sets[col])} unique values")

# Keep the most useful ones for later
esr_structures = value_sets.get(ESR_COL, set())
assets = value_sets.get(ASSETS_COL, set())

# Result
print("\nExample - linked ESR structures:")
print(esr_structures)


### 2.1.6 – Save the cleaned dataset
data.to_csv(CLEAN_FILE, sep=";", index=False)


#################################### 2.2 – DATA PROCESSING #########################

## Partnership organisation

### 2.2.1
def linked_startups(df, structure_name):
    """DataFrame of start-ups linked to an ESR structure (case-insensitive search)."""
    mask = df[ESR_COL].str.contains(structure_name, case=False, na=False)
    return df[mask]

#### Example
print("\nStart-ups linked to CNRS:")
cnrs = linked_startups(data, "Centre national de la recherche scientifique")
print(cnrs[["Raison sociale", "Statut", "Date création"]].head(8))


### 2.2.2
def partner_structures(df, structure_name):
    """Set of co-partner ESR structures (excluding the structure itself)."""
    linked = linked_startups(df, structure_name)
    if linked.empty:
        return set()
    partners = set()
    for cell in linked[ESR_COL].dropna():
        for s in cell.split(","):
            s = s.strip()
            if s and s != structure_name:
                partners.add(s)
    return partners

#### CNRS once again...
print("\nCNRS partners:")
cnrs_partners = partner_structures(data, "Centre national de la recherche scientifique")
print(sorted(cnrs_partners)[:15])


## Geographical location

### 2.2.3
def count_by_location(df, location_col):
    """Number of start-ups (unique SIREN) per location (Commune / Département / Région)."""
    if location_col not in df.columns:
        raise ValueError(f"Column '{location_col}' not found")
    return (df[["SIREN", location_col]]
            .drop_duplicates(subset="SIREN")[location_col]
            .value_counts())

#### Number of start-ups per region
print(count_by_location(data, "Région").head(10))

#### Number of start-ups per department
print(count_by_location(data, "Département").head(10))

#### Number of start-ups per municipality (most active ones)
print(count_by_location(data, "Commune").head(15))


### 2.2.4
def partner_locations(df, structure_name, column):
    """Number of start-ups partnered with a structure, grouped by location, sorted descending."""
    return count_by_location(linked_startups(df, structure_name), column)

#### Where are the CNRS-linked start-ups located (yes, CNRS again...)?
print("CNRS-linked start-ups by region:")
print(partner_locations(data, "Centre national de la recherche scientifique", "Région").head(10))

print("\nCNRS-linked start-ups by department:")
print(partner_locations(data, "Centre national de la recherche scientifique", "Département").head(10))

print("\nSorbonne Université-linked start-ups by municipality:")
print(partner_locations(data, "Sorbonne Université", "Commune").head(10))

print("\nHogwarts-linked start-ups by region:")
print(partner_locations(data, "Hogwarts", "Région"))  # Correctly returns an empty Series


## Optional plots

# Top 10 most active regions
count_by_location(data, "Région").head(10).plot(kind="bar", figsize=(10, 6))
plt.title("Number of start-ups by region")
plt.ylabel("Number of companies")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Geographical location of CNRS-linked start-ups
partner_locations(data, "Centre national de la recherche scientifique", "Région").head(10).plot(
    kind="bar", figsize=(10, 6), color="teal")
plt.title("CNRS-linked start-ups - Breakdown by region")
plt.ylabel("Number of start-ups")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


## Closed start-ups

### 2.2.5
def closed_startups(df):
    """Inactive start-ups + a 'lifetime' column in days."""
    closed = df[df["Statut"] == "inactive"].copy()
    closed["lifetime"] = (closed["Date fermeture"] - closed["Date création"]).dt.days
    # Drop negative durations (inconsistent data)
    return closed[closed["lifetime"] >= 0]


### 2.2.6
def closed_partners(df, structure_name):
    """Closed start-ups partnered with a structure, with their lifetime in days."""
    linked = linked_startups(df, structure_name)
    closed = linked[linked["Statut"] == "inactive"].copy()
    closed["lifetime"] = (closed["Date fermeture"] - closed["Date création"]).dt.days
    return closed


### 2.2.7 Closed start-ups by region
closed = closed_startups(data)
closed_by_region = (closed.groupby("Région", observed=True)
                    .size()
                    .sort_values(ascending=False))

print("Closed start-ups by region:")
print(closed_by_region.head(12))


### 2.2.8 Average lifetime
avg_lifetime = closed["lifetime"].mean()
print(f"\nOverall average lifetime: {avg_lifetime:.0f} days")

avg_lifetime_by_region = (closed.groupby("Région", observed=True)["lifetime"]
                          .mean()
                          .sort_values(ascending=False))

print("\nAverage lifetime by region (days):")
print(avg_lifetime_by_region.round(1).head(12))


## Assets

### 2.2.9
def with_asset(df, asset_type):
    """Companies owning at least one asset of the given type."""
    return df[df[ASSETS_COL].str.contains(asset_type, case=False, na=False)]


### 2.2.10
def count_by_category(df, column, categories):
    """Number of occurrences of each element of the set in the given column."""
    counts = {}
    for val in categories:
        counts[val] = df[column].str.contains(val, case=False, na=False).sum()
    return pd.Series(counts).sort_values(ascending=False)


### 2.2.11
def count_startups_with_asset(df, asset_type):
    """Number of companies owning at least one asset of this type."""
    return len(with_asset(df, asset_type))


### 2.2.12 Asset proportions
any_asset = data[data[ASSETS_COL].notna() & (data[ASSETS_COL] != "")]
total_with_asset = len(any_asset)

asset_shares = {}
for a in assets:
    n = count_startups_with_asset(data, a)
    asset_shares[a] = round((n / total_with_asset) * 100, 2) if total_with_asset else 0

print("\nShare of start-ups by asset type (%):")
print(pd.Series(asset_shares).sort_values(ascending=False))


### 2.2.13
def plot_asset_distribution(df):
    """Bar chart: share of companies by asset type."""
    pool = df[df[ASSETS_COL].notna() & (df[ASSETS_COL] != "")]
    total = len(pool)
    if total == 0:
        print("No asset recorded.")
        return
    unique_assets = value_set(df, ASSETS_COL)
    series = pd.Series({a: count_startups_with_asset(df, a) / total * 100 for a in unique_assets})
    series.sort_values().plot(kind="bar", color="cornflowerblue", figsize=(12, 7))
    plt.title("Share of companies by asset type\n(among those with at least one asset)")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Asset type")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


######################################### 2.3 – REPORT BY STRUCTURE ##############################

def structure_report(df, name):
    """Print a full report for a given ESR structure."""
    print(f"\n{'═' * 20} Report: {name} {'═' * 20}")

    linked = linked_startups(df, name)
    if linked.empty:
        print(f"No start-up found for '{name}'.")
        return

    print(f"Number of linked start-ups: {len(linked)}")

    # Location by department
    print("\nTop 10 departments:")
    print(partner_locations(df, name, "Département").head(10))

    # Closed start-ups
    f = closed_partners(df, name)
    print(f"\nOf which {len(f)} closed start-ups")
    if not f.empty:
        print(f"Average lifetime: {f['lifetime'].mean():.1f} days")
        print("\nAverage lifetime by region (closed):")
        print(f.groupby("Région", observed=True)["lifetime"]
              .mean().sort_values(ascending=False).round(1).head(10))

    # Assets
    if ASSETS_COL in linked.columns and linked[ASSETS_COL].notna().any():
        print("\nAsset distribution:")
        unique_assets = value_set(linked, ASSETS_COL)
        if unique_assets:
            print(count_by_category(linked, ASSETS_COL, unique_assets))
            plot_asset_distribution(linked)
        else:
            print("No asset recorded.")
    else:
        print("No asset information.")


######################## EXTENSION: GEOGRAPHICAL MAP (geopandas + contextily) ##########################

def load_france_map():
    """Load the GeoJSON of metropolitan French regions, projected to EPSG:3857."""
    url = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    gdf = gpd.read_file(url)
    # Drop the overseas regions, otherwise the map gets squashed
    overseas = ["Guadeloupe", "Martinique", "Guyane", "La Réunion", "Mayotte"]
    gdf = gdf[~gdf["nom"].isin(overseas)][["nom", "geometry"]]
    gdf = gdf.rename(columns={"nom": "Région"})
    return gdf.to_crs(epsg=3857)


def map_by_region(df, title="Number of start-ups by region"):
    """Choropleth map of the number of start-ups per region."""
    counts = count_by_location(df, "Région")
    gdf = load_france_map()
    gdf = gdf.merge(counts.rename("count"), left_on="Région",
                    right_index=True, how="left")
    gdf["count"] = gdf["count"].fillna(0)

    fig, ax = plt.subplots(figsize=(14, 12))
    gdf.plot(column="count", cmap="viridis", linewidth=1, edgecolor="white",
             legend=True, legend_kwds={"shrink": 0.6}, alpha=0.9, ax=ax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager, crs=gdf.crs)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


########################################### INTERACTIVE MENU ###################################

def show_menu():
    print("\n" + "═" * 70)
    print("  STUDY OF START-UPS LINKED TO FRENCH HIGHER EDUCATION & RESEARCH")
    print("═" * 70)
    print("  1. List ESR structures")
    print("  2. Full report for a structure")
    print("  3. Top regions / departments / municipalities")
    print("  4. Overall asset distribution (chart)")
    print("  5. Average lifetime of closed start-ups")
    print("  6. Map: all start-ups by region")
    print("  7. Map: start-ups linked to an ESR structure")
    print("  8. Quit")
    return input("\nChoice (1-8) → ").strip()


def main(df):
    while True:
        choice = show_menu()

        if choice == "1":
            structs = sorted(value_set(df, ESR_COL))
            print(f"\n{len(structs)} ESR structures:")
            for i, s in enumerate(structs, 1):
                print(f"  {i:3d}. {s}")

        elif choice == "2":
            name = input("\nStructure name (or part of it) → ").strip()
            if name:
                structure_report(df, name)

        elif choice == "3":
            for level, n in [("Région", 10), ("Département", 10), ("Commune", 15)]:
                print(f"\nTop {n} – {level}:")
                print(count_by_location(df, level).head(n))

        elif choice == "4":
            plot_asset_distribution(df)

        elif choice == "5":
            f = closed_startups(df)
            print(f"\nOverall average lifetime: {f['lifetime'].mean():.0f} days")
            print("\nBy region:")
            print(f.groupby("Région", observed=True)["lifetime"]
                  .mean().sort_values(ascending=False).round(1).head(12))

        elif choice == "6":
            print("Generating map...")
            map_by_region(df)

        elif choice == "7":
            name = input("\nStructure name (or part of it) → ").strip()
            if name:
                linked = linked_startups(df, name)
                if linked.empty:
                    print(f"No start-up found for '{name}'.")
                else:
                    map_by_region(linked, f"Start-ups linked to '{name}' by region")
            else:
                print("Cancelled.")

        elif choice == "8":
            print("End of program.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main(data)
