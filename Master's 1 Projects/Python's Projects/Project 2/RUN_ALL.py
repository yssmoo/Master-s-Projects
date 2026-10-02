"""
                                        Project 2 - Open Food Facts
                                    Abdellah Nait Akli / Yassir Motia
                                                  M1 ESA
                                               Mrs. BROSSE
"""


""" Package imports """


import requests  # To interact with the API
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


# 100 products per page is the maximum allowed by the API
PAGE_SIZE = 100
# Capped at 20 pages to avoid waiting for hours
MAX_PAGES = 20

# Open Food Facts API endpoint (v2)
url_api_OFF = "https://world.openfoodfacts.org/api/v2/search"


#############################################################################################################
#           Part 1: Data preparation — the chocolate-hazelnut spread example

## Question 1: df_spreads

def get_spreads_page1():

    """     Start by defining the request parameters -> these are described
            in the API documentation: https://wiki.openfoodfacts.org/API/Read/Search   """

    # Only 3 columns are kept, as required by the assignment
    # (the search term stays in French: it is the Open Food Facts category name)
    parameters = {"search_terms": "Pâte à tartiner chocolat et noisettes",
                  "search_simple": 1,
                  "action": "process",
                  "json": 1,
                  "page_size": PAGE_SIZE,
                  "page": 1,
                  "fields": "code,categories_tags_fr,nutrition_grades"}

    req = requests.get(url_api_OFF, params=parameters)
    data = req.json()

    df = pd.DataFrame(data.get("products", []))
    return df

"""To display and inspect the DataFrame:"""
# df_spreads = get_spreads_page1()
# print(df_spreads)

## Question 2: df_spreads_full

def get_spreads_full():

    all_products = []
    page = 1

    # Loop over pages until there are none left
    while page <= MAX_PAGES:
        parameters = {"search_terms": "pate a tartiner chocolat noisettes",
                      "page_size": PAGE_SIZE,
                      "page": page,
                      "fields": "code,categories_tags_fr,nutriscore_grade"}

        req = requests.get(url_api_OFF, params=parameters)
        if req.status_code != 200:  # Stop if the server responds badly (e.g. 404 -> page not found, 500 -> server error)
            break
        products = req.json().get("products", [])

        if not products:  # No more products = everything has been retrieved
            break

        all_products.extend(products)
        page += 1
        time.sleep(0.2)  # Short pause to avoid hammering the API
    df_spreads_full = pd.DataFrame(all_products)
    return df_spreads_full

"""To display the DataFrame:"""
# df_spreads_full = get_spreads_full()
# print(df_spreads_full)

###############################################################################################################
#                                 Part 2: Data processing

"""First, clean the dataset"""

def clean_dataframe(df):

    """Turn the raw API output into an organised DataFrame."""

    df = df.copy()

    # Depending on the request, the grade comes back as 'nutriscore_grade' or 'nutrition_grades'
    if "nutriscore_grade" in df.columns and "nutrition_grades" not in df.columns:
        df = df.rename(columns={"nutriscore_grade": "nutrition_grades"})

    if "nutrition_grades" not in df.columns:
        df["nutrition_grades"] = None

    # Replace empty strings with NaN before filtering
    df["nutrition_grades"] = df["nutrition_grades"].replace("", np.nan)
    df = df.dropna(subset=["nutrition_grades"])
    # Keep only valid grades a, b, c, d, e
    df = df[df["nutrition_grades"].str.lower().isin(["a", "b", "c", "d", "e"])]
    df["nutrition_grades"] = df["nutrition_grades"].str.lower()

    print(f"Products with a valid Nutri-Score: {len(df)}")

    # additives_tags is a list, we simply count its length
    if "additives_tags" not in df.columns:
        df["additives_tags"] = [[] for _ in range(len(df))]
    df["n_additives"] = df["additives_tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Extract the nutrient values we are interested in
    if "nutriments" not in df.columns:
        df["nutriments"] = [{} for _ in range(len(df))]
    nutrients = ["sugars_100g", "fat_100g", "salt_100g", "energy-kcal_100g"]
    names = ["sugar", "fat", "salt", "energy"]

    for col, name in zip(nutrients, names):
        df[name] = df["nutriments"].apply(lambda x, c=col: x.get(c) if isinstance(x, dict) else None)
        df[name] = pd.to_numeric(df[name], errors="coerce")

    return df


## Question 3: get_food_dataframe

def get_food_dataframe(food_type):

    all_products = []
    page = 1

    while page <= MAX_PAGES:
        # This time we also retrieve additives and nutrients (not just the Nutri-Score)
        parameters = {"search_terms": food_type,
                      "page_size": PAGE_SIZE,
                      "page": page,
                      "fields": "code,categories_tags_fr,nutriscore_grade,additives_tags,nutriments"}
        req = requests.get(url_api_OFF, params=parameters)
        if req.status_code != 200:
            break
        data_json = req.json()
        products = data_json.get("products", [])
        if not products:
            break
        all_products.extend(products)
        page += 1
        time.sleep(0.3)

    return pd.DataFrame(all_products)



## Question 4: count by Nutri-Score

def count_nutriscore(df):

    result = {}
    for grade in df["nutrition_grades"]:
        if grade not in result:
            result[grade] = 0
        result[grade] += 1

    df_result = pd.DataFrame(list(result.items()), columns=["nutriscore", "count"])
    df_result = df_result.sort_values("nutriscore").reset_index(drop=True)

    return df_result


## Question 5: additives per product

def count_additives_per_product(df):

    codes = []
    n_additives = []

    for i in range(len(df)):
        codes.append(df["code"].iloc[i])
        additives = df["additives_tags"].iloc[i]
        # If it is a list, take its length; otherwise 0
        if isinstance(additives, list):
            n_additives.append(len(additives))
        else:
            n_additives.append(0)

    df = pd.DataFrame({"code": codes, "n_additives": n_additives})
    return df


## Question 6: additional analyses

def global_statistics(df):

    """Average nutritional statistics."""

    res = pd.DataFrame({"Avg sugar (g/100g)": [round(df["sugar"].mean(), 2)],
                        "Avg fat (g/100g)": [round(df["fat"].mean(), 2)],
                        "Avg salt (g/100g)": [round(df["salt"].mean(), 2)],
                        "Avg additives": [round(df["n_additives"].mean(), 2)],
                        "Avg energy (kcal)": [round(df["energy"].mean(), 2)]})

    return res



def nutrition_correlation(df):

    """Correlation matrix between nutritional variables."""

    cor = df[["sugar", "fat", "salt", "n_additives"]].corr()

    return cor


#                                                       Part 3: Presentation of results

## Question 7: Nutri-Score pie chart

def pie_nutriscore(df):

    counts = df["nutrition_grades"].value_counts().sort_index()

    # Map each Nutri-Score letter to a colour
    colors = {"a": "darkgreen", "b": "yellowgreen", "c": "gold", "d": "orange", "e": "crimson"}
    clrs = [colors.get(k, "grey") for k in counts.index]
    plt.figure(figsize=(6, 6))
    # autopct prints the percentage on each slice
    plt.pie(counts, labels=counts.index.str.upper(), autopct='%1.1f%%', colors=clrs, startangle=90)
    plt.title("Nutri-Score distribution")
    plt.tight_layout()
    plt.show()


## Question 8: additives by Nutri-Score

def boxplot_additives(df):

    # Sort the grades so the boxes appear in order a → e
    ordered = sorted(df["nutrition_grades"].dropna().unique())
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="nutrition_grades", y="n_additives", data=df, order=ordered)
    plt.title("Number of additives by Nutri-Score")
    plt.xlabel("Nutri-Score")
    plt.ylabel("Number of additives")
    plt.tight_layout()
    plt.show()


## Question 9: other charts


def histogram_sugar(df):

    """Plot a histogram of sugar content."""

    plt.figure(figsize=(8, 5))
    plt.hist(df["sugar"].dropna(), bins=30, color="steelblue", edgecolor="white")
    plt.title("Sugar distribution (g/100g)")
    plt.xlabel("Sugar (g/100g)")
    plt.ylabel("Number of products")
    plt.tight_layout()
    plt.show()


def heatmap_correlation(df):

    """Plot the correlation matrix as a heatmap."""

    corr = nutrition_correlation(df)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation between nutritional variables")
    plt.tight_layout()
    plt.show()


def bar_nutriscore(df):

    """Bar chart showing the number of products for each Nutri-Score letter."""

    counts = df["nutrition_grades"].value_counts().sort_index()
    if counts.empty:
        print("No product with an available Nutri-Score.")
        return
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="steelblue", edgecolor="white")
    plt.title("Number of products by Nutri-Score")
    plt.xlabel("Nutri-Score")
    plt.ylabel("Number of products")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# EXTENSION: controversial additives
# Source: public list on GitHub
# https://raw.githubusercontent.com/nicklockwood/FoodAdditives/master/FoodAdditives.json

url_github = "https://raw.githubusercontent.com/nicklockwood/FoodAdditives/master/FoodAdditives.json"


# Fallback list if the URL is not reachable
ADDITIVES_FALLBACK = {"e211": "Sodium benzoate",
                      "e220": "Sulphur dioxide",
                      "e249": "Potassium nitrite",
                      "e250": "Sodium nitrite",
                      "e251": "Sodium nitrate",
                      "e252": "Potassium nitrate",
                      "e407": "Carrageenan",
                      "e476": "PGPR",
                      "e621": "Monosodium glutamate"}



def load_controversial_additives():

    try:
        req = requests.get(url_github, timeout=5)
        if req.status_code == 200:
            data = req.json()
            additives = {}
            for item in data:
                # Codes look like "E211", normalise them to lowercase -> "e211"
                code = "e" + str(item.get("number", "")).lower().replace("e", "")
                name = item.get("name", "")
                additives[code] = name
            print(f"  {len(additives)} additives loaded from GitHub.")
            return additives

    except Exception:
        pass

    # If GitHub is not reachable, use the manual list
    print("  Could not reach GitHub, using the fallback list.")
    return ADDITIVES_FALLBACK


def count_controversial_additives(df):

    """Add a column with the number of controversial additives per product."""

    risky_additives = load_controversial_additives()
    df = df.copy()

    def count(additives):
        if not isinstance(additives, list):
            return 0
        return sum(1 for a in additives if a.replace("en:", "").replace("fr:", "") in risky_additives)

    df["n_controversial_additives"] = df["additives_tags"].apply(count)

    return df


def analyse_controversial_additives(df):

    """Average number of controversial additives by Nutri-Score."""

    return df.groupby("nutrition_grades")["n_controversial_additives"].mean().round(2)


def plot_controversial_additives(df):

    data = analyse_controversial_additives(df)
    if data.empty:
        print("No data available on controversial additives.")
        return
    plt.figure(figsize=(8, 5))
    data.plot(kind="bar", color="tomato", edgecolor="white")
    plt.title("Average controversial additives by Nutri-Score")
    plt.xlabel("Nutri-Score")
    plt.ylabel("Average number of controversial additives")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# COMPARISON BETWEEN 2 PRODUCTS

def compare_products(p1, p2):

    print(f"\nDownloading « {p1} »...")
    df1 = clean_dataframe(get_food_dataframe(p1))

    print(f"Downloading « {p2} »...")
    df2 = clean_dataframe(get_food_dataframe(p2))

    data = pd.DataFrame({p1: [df1["sugar"].mean(), df1["n_additives"].mean()],
                         p2: [df2["sugar"].mean(), df2["n_additives"].mean()]},
                        index=["Avg sugar (g/100g)", "Avg additives"])

    data.T.plot(kind="bar")
    plt.title("Nutritional comparison")
    plt.ylabel("Average value")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()



########################################################################################################################
# USER MENU (Question 10)

# Search terms sent to the API (kept in French where it gives better matches on Open Food Facts)
available_products = ["pizza",
                      "cereales",
                      "biscuits",
                      "jus de fruits",
                      "pate a tartiner chocolat et noisettes",
                      "yaourt",
                      "chips",
                      "chocolate"]



def menu():

    """Interactive menu to choose the food type to analyse."""

    print("\n=== Open Food Facts Analysis ===")
    print("\nAvailable products:")
    for i, p in enumerate(available_products, 1):
        print(f"  {i} - {p}")
    print("  0 - Enter a product manually")

    choice = input("\nYour choice (number): ").strip()

    if choice == "0":
        return input("Enter the food type: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(available_products):
            return available_products[idx]
    except ValueError:
        pass

    print("Invalid choice, enter a number between 0 and", len(available_products))
    return menu()



# MAIN PROGRAM

def main():
    product = menu()

    print(f"\nDownloading data for « {product} »...\n")
    df = get_food_dataframe(product)
    df = clean_dataframe(df)

    if df.empty:
        print("No product with a valid Nutri-Score was found. Try another search term.")
        return

    df = count_controversial_additives(df)

    print(f"\nNumber of products analysed: {len(df)}")

    print("\n--- Nutri-Score distribution ---")
    print(count_nutriscore(df).to_string(index=False))

    print("\n--- Additives per product (first 5) ---")
    print(count_additives_per_product(df).head().to_string(index=False))

    print("\n--- Global statistics ---")
    print(global_statistics(df).to_string(index=False))

    print("\n--- Controversial additives by Nutri-Score ---")
    print(analyse_controversial_additives(df))

    pie_nutriscore(df)
    bar_nutriscore(df)
    boxplot_additives(df)
    histogram_sugar(df)
    heatmap_correlation(df)
    plot_controversial_additives(df)

    comp = input("\nCompare with another product? (y/n): ").strip().lower()
    if comp == "y":
        p2 = input("Enter the second product: ").strip()
        compare_products(product, p2)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
