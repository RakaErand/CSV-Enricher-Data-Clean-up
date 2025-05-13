import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO
from datetime import datetime


def split_csv(df, chunk_size=100):
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


def create_zip_from_chunks(chunks, base_name='chunk', store=None, store_address=None, original_columns=None):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, chunk in enumerate(chunks):
            if "Pool Store" in chunk.columns:
                chunk["Pool Store"] = store
            if "Pool Store Address" in chunk.columns:
                chunk["Pool Store Address"] = store_address

            if original_columns:
                chunk = chunk[original_columns]

            file_buffer = BytesIO()
            chunk.to_csv(file_buffer, index=False)
            file_buffer.seek(0)
            zip_file.writestr(f"{base_name}_{idx + 1}.csv", file_buffer.read())

    zip_buffer.seek(0)
    return zip_buffer


def normalize(val):
    return str(val).strip().lower() if pd.notna(val) else ""


def validate_pool_volume(val):
    if pd.isna(val):
        return None
    try:
        float_val = float(str(val).strip())
        return int(float_val) if float_val.is_integer() else float_val
    except (ValueError, TypeError):
        return None


def get_signature(row):
    first_name = normalize(
        next((row.get(f) for f in ["first_name", "FirstName"] if f in row), ""))
    if not first_name:
        return None

    last_name = normalize(next((row.get(f)
                          for f in ["last_name", "LastName"] if f in row), ""))

    water_type = normalize(
        next((row.get(f) for f in ["water_type", "WaterType"] if f in row), ""))

    pool_volume = validate_pool_volume(
        next((row.get(f) for f in ["poolVolume", "PoolVolume"] if f in row), None))

    if last_name:
        key = (first_name, last_name, water_type, pool_volume)
    else:
        key = (first_name, water_type, pool_volume)

    helper_fields = [
        ("city", "City"),
        ("state", "State"),
        ("address", "Addess1"),
    ]

    helpers = []
    for field_opts in helper_fields:
        value = next((row.get(f) for f in field_opts if f in row), "")
        helpers.append(normalize(value))

    return (key, tuple(helpers))


def enrich_target_with_phones(clean_df, target_df):
    phone_filled = 0
    clean_lookup = {}

    for _, row in clean_df.iterrows():
        sig = get_signature(row)
        phone = str(row.get("phone", "")).strip()
        if phone:
            clean_lookup[sig] = phone

    for idx, row in target_df.iterrows():
        phone = str(row.get("PhoneNumber", "")).strip()
        if not phone:
            sig = get_signature(row)
            new_phone = clean_lookup.get(sig)
            if new_phone:
                target_df.at[idx, "PhoneNumber"] = new_phone
                phone_filled += 1

    st.info(f"📞 {phone_filled} missing phone numbers filled in target file.")
    return target_df


def filter_bak_using_test_flags(bak_df, clean_df):
    clean_df = clean_df.copy()
    bak_df = bak_df.copy()

    clean_df['has_test_taken'] = clean_df['has_test_taken'].astype(
        str).str.strip().str.lower()
    clean_df['has_test_taken'] = clean_df['has_test_taken'].isin(
        ['true', '1', 'yes'])

    tested_keys = set()
    tested_users = clean_df[clean_df['has_test_taken']
                            ].iloc[1:]
    for _, row in tested_users.iterrows():
        sig = get_signature(row)
        if sig is not None and sig[0]:
            tested_keys.add(sig[0])

    filtered_rows = []
    for _, row in bak_df.iterrows():
        sig = get_signature(row)
        if sig is None:
            filtered_rows.append(row.to_dict())
            continue

        key = sig[0]
        if key not in tested_keys:
            filtered_rows.append(row.to_dict())

    return pd.DataFrame(filtered_rows), tested_keys


def identify_reimport_candidates(clean_df, bak_signatures):
    reimport_rows = []
    for _, row in clean_df.iterrows():
        sig = get_signature(row)
        if sig is None:
            continue
        new_row = row.copy()

        if sig not in bak_signatures:
            new_row["needs_chemical_profile_update"] = "No"
            new_row["eligible_for_reimport"] = "Yes"
        else:
            new_row["needs_chemical_profile_update"] = "Yes"
            new_row["eligible_for_reimport"] = "No"

        reimport_rows.append(new_row)

    return pd.DataFrame(reimport_rows)


def add_update_flags(bak_df, tested_keys):
    flagged_rows = []
    for _, row in bak_df.iterrows():
        sig = get_signature(row)
        if sig is None:
            flagged_rows.append(row)
            continue

        key = sig[0]
        if key not in tested_keys:
            new_row = row.copy()
            new_row["needs_chemical_profile_update"] = "No"
            new_row["eligible_for_reimport"] = "Yes"
            flagged_rows.append(new_row)
        else:
            flagged_rows.append(row)
    return pd.DataFrame(flagged_rows)


st.set_page_config(page_title="📂 CSV Enricher", layout="centered")
st.markdown("### 📥 Upload CSV Files")

uploaded_files = st.file_uploader("Upload 2 CSV files (1 with has_test_taken, 1 target .bak)", type=[
    'csv'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 2:
        st.warning("Please upload exactly two CSV files.")
    else:
        file1, file2 = uploaded_files
        df1 = pd.read_csv(file1, sep=None, engine='python')
        df2 = pd.read_csv(file2, sep=None, engine='python')

        df2.columns = [col.replace(".1", "") for col in df2.columns]

        if "has_test_taken" in df1.columns:
            clean_df, bak_df = df1, df2
        elif "has_test_taken" in df2.columns:
            clean_df, bak_df = df2, df1
        else:
            st.error("❌ Neither file contains a 'has_test_taken' column.")
            st.stop()

        clean_df = clean_df.reset_index(drop=True)
        bak_df = bak_df.reset_index(drop=True)

        st.success(f"📄 .bak file loaded: {len(bak_df)} rows")
        st.success(f"✅ Clean file loaded: {len(clean_df)} rows")

        bak_filtered, tested_keys = filter_bak_using_test_flags(
            bak_df, clean_df)
        removed_count = len(bak_df) - len(bak_filtered)
        st.info(
            f"🧹 Removed {removed_count} test takers from .bak")

        flagged_bak = add_update_flags(bak_filtered, tested_keys)

        bak_signatures = set(get_signature(row)
                             for _, row in bak_df.iterrows())

        reimport_df = identify_reimport_candidates(clean_df, bak_signatures)

        final_df = enrich_target_with_phones(clean_df, flagged_bak)

        st.write("✅ Final Flagged + Enriched Data Preview", final_df.head())

        if st.button("Download Cleaned CSV"):
            now = datetime.now().strftime("%Y%m%d_%H%M")
            output_filename = f"cleaned_flagged_customers_{now}.csv"
            output_csv = final_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Flagged CSV",
                data=output_csv,
                file_name=output_filename,
                mime="text/csv"
            )
