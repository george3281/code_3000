import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records, where the
    (age, gender, zip3) combination appears exactly once
    in each dataset.
    """
    # Define the quasi-identifiers used for linkage
    qi_cols = ["age", "zip3", "gender"]

    # Count how often each quasi-identifier combination appears
    anon_counts = (
        anon_df.groupby(qi_cols)
        .size()
        .reset_index(name="anon_count")
    )
    aux_counts = (
        aux_df.groupby(qi_cols)
        .size()
        .reset_index(name="aux_count")
    )

    # Keep only combinations that appear exactly once in each dataset
    unique_qi = anon_counts.merge(aux_counts, on=qi_cols, how="inner")
    unique_qi = unique_qi[
        (unique_qi["anon_count"] == 1) & (unique_qi["aux_count"] == 1)
    ][qi_cols]

    # Filter original dataframes to those unique quasi-identifiers
    anon_unique = anon_df.merge(unique_qi, on=qi_cols, how="inner")
    aux_unique = aux_df.merge(unique_qi, on=qi_cols, how="inner")

    # Perform one-to-one merge and select requested columns
    merged = anon_unique.merge(aux_unique, on=qi_cols, suffixes=("_anon", "_aux"))
    result = merged[["anon_id", "name"]].rename(columns={"name": "matched_name"})
    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if anon_df.empty:
        return 0.0

    uniquely_identified = matches_df["anon_id"].nunique()
    total_anon = len(anon_df)
    return uniquely_identified / total_anon
