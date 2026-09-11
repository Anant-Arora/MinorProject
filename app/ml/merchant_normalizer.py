import pandas as pd
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_csv("data/synthetic/transactions.csv")

# ---------------------------------------------------------
# STEP 1: Clean each description down to just the merchant
# portion, stripping reference numbers and common UPI/bank
# boilerplate tokens that appear across every merchant and
# would otherwise create false similarity between unrelated
# merchants.
# ---------------------------------------------------------
BOILERPLATE_TOKENS = [
    "upi", "upiintent", "intent", "hdfc", "icici", "axis", "sbi",
    "indusind", "airtel", "nsdl", "pvt", "ltd", "private", "limited",
    "technologies", "technology", "company", "enterprises", "solutions",
    "services", "commerce", "corporation", "corp", "inc",
]

def clean_merchant_text(description):
    # Remove trailing reference number
    text = re.sub(r'/\d{10,}/?$', '', description)
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = text.lower()
    # Remove boilerplate tokens
    tokens = [t for t in text.split() if t not in BOILERPLATE_TOKENS]
    return " ".join(tokens).strip()

df["merchant_clean_text"] = df["description"].apply(clean_merchant_text)

# ---------------------------------------------------------
# STEP 2: Work on UNIQUE cleaned merchant strings only
# ---------------------------------------------------------
unique_merchants = df["merchant_clean_text"].unique().tolist()

# ---------------------------------------------------------
# STEP 3: Vectorize using character n-grams (captures
# substring overlap like "swiggy" inside "swiggy blr123",
# which word-level matching would miss) and compute
# pairwise cosine similarity.
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
tfidf_matrix = vectorizer.fit_transform(unique_merchants)
similarity_matrix = cosine_similarity(tfidf_matrix)

# ---------------------------------------------------------
# STEP 4: Build a graph where an edge connects two merchant
# strings if their similarity is above a threshold. Merchant
# clusters = connected components of this graph.
# ---------------------------------------------------------
SIMILARITY_THRESHOLD = 0.3

graph = nx.Graph()
graph.add_nodes_from(unique_merchants)

for i in range(len(unique_merchants)):
    for j in range(i + 1, len(unique_merchants)):
        if similarity_matrix[i, j] >= SIMILARITY_THRESHOLD:
            graph.add_edge(unique_merchants[i], unique_merchants[j])

clusters = list(nx.connected_components(graph))

# ---------------------------------------------------------
# STEP 5: Known alias lookup table. Pure text similarity
# CANNOT merge a merchant's legal/registered company name
# with its consumer brand name (e.g. "Bundl" vs "Swiggy")
# since they share no substring. Production systems handle
# this with a maintained alias table, not ML guesswork.
# ---------------------------------------------------------
KNOWN_ALIASES = {
    "bundl": "swiggy",
    "blink": "blinkit",
}

def apply_known_aliases(merchant_text):
    for alias, canonical in KNOWN_ALIASES.items():
        if alias in merchant_text:
            return canonical
    return None

# ---------------------------------------------------------
# STEP 6: Assign each cluster a canonical label. If any
# member matches a known alias, use that; otherwise use
# the shortest string in the cluster as a representative.
# ---------------------------------------------------------
merchant_to_canonical = {}

for cluster in clusters:
    alias_hit = None
    for member in cluster:
        alias_hit = apply_known_aliases(member)
        if alias_hit:
            break

    if alias_hit:
        canonical_label = alias_hit
    else:
        canonical_label = min(cluster, key=len)

    for member in cluster:
        merchant_to_canonical[member] = canonical_label

for merchant_text in unique_merchants:
    alias_hit = apply_known_aliases(merchant_text)
    if alias_hit:
        merchant_to_canonical[merchant_text] = alias_hit

df["predicted_canonical_merchant"] = df["merchant_clean_text"].map(merchant_to_canonical)

# ---------------------------------------------------------
# STEP 7: Final harmonization pass. Clustering and alias
# lookup can independently produce two different labels for
# the SAME real merchant. This pass merges any predicted
# labels sharing a known brand keyword into one label.
# ---------------------------------------------------------
BRAND_KEYWORDS = ["swiggy", "zomato", "blinkit", "netflix", "spotify", "hotstar", "amazon"]

def harmonize_label(label):
    for keyword in BRAND_KEYWORDS:
        if keyword in label:
            return keyword
    return label

df["predicted_canonical_merchant"] = df["predicted_canonical_merchant"].apply(harmonize_label)

# ---------------------------------------------------------
# Show results
# ---------------------------------------------------------
print(f"Total unique raw merchant strings: {len(unique_merchants)}")
print(f"Total predicted canonical merchants: {df['predicted_canonical_merchant'].nunique()}")
print()
print(df[["description", "merchant_clean_text", "predicted_canonical_merchant"]].drop_duplicates(subset="merchant_clean_text").to_string(index=False))

df.to_csv("data/synthetic/transactions_normalized.csv", index=False)
print("\nSaved to: data/synthetic/transactions_normalized.csv")