import os
import pandas as pd
import re

def parse_bank_statement(file_path: str, user_id: str = "user_1") -> pd.DataFrame:
    """
    Ingests a raw real-world Indian bank or UPI CSV export, 
    dynamically maps schemas, cleans currency/dates, filters expenses, 
    and strips heavy UPI boilerplate text for ML compatibility.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Statement file not found at: {file_path}")
        
    raw_df = pd.read_csv(file_path)
    columns_lower = [c.lower().strip() for c in raw_df.columns]
    
    mapped_rows = []
    
    is_hdfc = any('narration' in c for c in columns_lower)
    is_upi_export = any('transaction id' in c or 'note' in c for c in columns_lower)
    
    for _, row in raw_df.iterrows():
        date_str, description, amount = None, None, 0.0
        
        if is_hdfc:
            date_str = str(row.get('Date', row.get('Txn Date', ''))).strip()
            raw_desc = str(row.get('Narration', row.get('Description', ''))).strip()
            withdrawal = row.get('Withdrawal Amt.', row.get('Debit', 0))
            amount = parse_amount(withdrawal)
            description = clean_upi_description(raw_desc)
            
        elif is_upi_export:
            date_str = str(row.get('Date', row.get('Timestamp', ''))).strip()
            raw_desc = str(row.get('Note', row.get('Description', row.get('Category', '')))).strip()
            raw_amt = row.get('Amount', 0)
            txn_type = str(row.get('Type', row.get('Flow', 'DEBIT'))).upper()
            
            parsed_amt = parse_amount(raw_amt)
            if 'CREDIT' in txn_type or 'RECEIVED' in txn_type:
                amount = 0.0
            else:
                amount = abs(parsed_amt)
            description = clean_upi_description(raw_desc)
        else:
            date_str = str(row.get('Date', row.get('date', ''))).strip()
            raw_desc = str(row.get('Description', row.get('Narration', row.get('Merchant', '')))).strip()
            raw_amt = row.get('Amount', row.get('Withdrawal', 0))
            amount = parse_amount(raw_amt)
            description = clean_upi_description(raw_desc)

        if not date_str or pd.isna(date_str) or amount <= 0:
            continue
            
        normalized_date = clean_date(date_str)
        if not normalized_date:
            continue

        mapped_rows.append({
            'user_id': user_id,
            'date': normalized_date,
            'description': description,
            'amount': amount,
            'is_subscription': 0,
            'is_price_creep': 0,
            'is_anomaly': 0
        })
        
    cleaned_df = pd.DataFrame(mapped_rows)
    print(f"[INFO] Successfully parsed and cleaned {len(cleaned_df)} valid expense rows.")
    return cleaned_df


def clean_upi_description(raw_desc: str) -> str:
    """
    Strips Indian UPI boilerplate tokens, transaction IDs, and bank handles.
    Example: 'UPI/DR/123456789012/Zomato Media Pvt/paytm/hdfcbank' -> 'Zomato Media Pvt'
    """
    if not raw_desc or pd.isna(raw_desc):
        return "UNKNOWN"
        
    text = str(raw_desc).strip()
    
    # If it's a standard UPI transfer string containing slashes
    if "UPI/" in text.upper() or text.count('/') >= 3:
        parts = text.split('/')
        # Usually, the merchant/receiver name is situated near the middle segments
        # Let's filter out known boilerplate tokens (UPI, DR, CR, reference numbers, bank handles)
        boilerplate_tokens = {'upi', 'dr', 'cr', 'paytm', 'gpay', 'phonepe', 'ybl', 'okicici', 'oksbi', 'okhdfcbank', 'hdfcbank', 'sbi', 'icici'}
        
        valid_parts = []
        for part in parts:
            p_clean = part.strip()
            # Skip if it's purely digits (reference numbers) or a known boilerplate keyword
            if p_clean.isdigit() or len(p_clean) <= 2 or p_clean.lower() in boilerplate_tokens:
                continue
            valid_parts.append(p_clean)
            
        if valid_parts:
            # Return the most likely candidate part (e.g., the merchant name)
            return valid_parts[0]
            
    # Fallback: remove 10+ digit reference numbers if present anywhere else
    cleaned = re.sub(r'/\d{10,}/?$', '', text)
    return cleaned.strip()


def parse_amount(val) -> float:
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace('Rs.', '').replace('INR', '').replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def clean_date(date_str: str) -> str:
    try:
        dt = pd.to_datetime(date_str, dayfirst=True)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return None