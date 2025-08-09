# data/processor.py
import pandas as pd
import ast

def process_data(file_path: str, sheet_name: str = 'Data', header_row: int = 1) -> pd.DataFrame:
    """Clean and process the Excel data for agent consumption."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        print("Excel file loaded successfully.")

        # Clean column names
        df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\xa0', ' ')
        
        # Correct column mapping based on actual dataset structure
        column_mapping = {
            'User Name': 'Sales Rep Name',
            'Business Name': 'Prospect Business Name', 
            'Website URL': 'Website',
            'Phone': 'Prospect Phone',
            'Email': 'Prospect Email',
            'Category - Primary': 'Primary Category',
            'Category - Secondary': 'Secondary Category',
            'All Signals/SMB Data Points': 'BuzzBoard Data'
        }

        # Apply mapping only for columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

        # Forward fill essential columns
        essential_columns = ['Customer', 'Sales Rep Name']
        for col in essential_columns:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Remove header/separator rows
        if 'Prospect Business Name' in df.columns:
            df = df[df['Prospect Business Name'] != 'SMB'].copy()
            df = df.dropna(subset=['Prospect Business Name']).copy()

        # Process digital signals with simplified parsing
        def parse_signals(data_string):
            """Parse digital signals data."""
            if pd.isna(data_string) or str(data_string).strip() == '':
                return {}
            
            try:
                if isinstance(data_string, dict):
                    return data_string
                    
                if isinstance(data_string, str):
                    cleaned = data_string.strip()
                    if cleaned.startswith('[') or cleaned.startswith('{'):
                        try:
                            parsed = ast.literal_eval(cleaned)
                            if isinstance(parsed, list) and parsed:
                                return parsed[0] if isinstance(parsed[0], dict) else {}
                            elif isinstance(parsed, dict):
                                return parsed
                        except:
                            pass
                    
                    # Manual key:value parsing
                    signals = {}
                    if ':' in cleaned:
                        pairs = cleaned.replace(';', ',').replace('|', ',').split(',')
                        for pair in pairs:
                            if ':' in pair:
                                key, value = pair.split(':', 1)
                                signals[key.strip()] = value.strip()
                    return signals
                        
                return {}
            except Exception:
                return {}

        # Process BuzzBoard data if exists
        if 'BuzzBoard Data' in df.columns:
            df['BuzzBoard Data Parsed'] = df['BuzzBoard Data'].apply(parse_signals)
        else:
            df['BuzzBoard Data Parsed'] = [{}] * len(df)

        # Standardize digital signal columns
        def get_signal_value(signals_dict, key_variations):
            """Extract signal value with multiple key options."""
            if not isinstance(signals_dict, dict):
                return 'No'
            
            for key in key_variations:
                if key in signals_dict:
                    value = str(signals_dict[key]).lower()
                    if value in ['yes', 'true', '1', 'active']:
                        return 'Yes'
                    elif value in ['no', 'false', '0', 'inactive']:
                        return 'No'
                    return signals_dict[key]
            return 'No'

        # Create standardized digital signal columns
        signal_mappings = {
            'SEM': ['SEM', 'Google Ads', 'paid_search', 'google_ads'],
            'FB latest_posts': ['FB latest_posts', 'Facebook', 'social_media', 'FB_posts'],
            'Reviews (local and social)': ['Reviews (local and social)', 'Reviews', 'rating'],
            'Display ads': ['Display ads', 'display_ads', 'banner_ads'],
            'Google Places': ['Google Places', 'GMB', 'local_listing', 'google_my_business']
        }
        
        # First check if columns already exist in dataset
        for signal_col in signal_mappings.keys():
            if signal_col not in df.columns:
                df[signal_col] = df['BuzzBoard Data Parsed'].apply(
                    lambda x: get_signal_value(x, signal_mappings[signal_col])
                )

        # Process competitor data
        for i in range(1, 4):
            comp_col = f'Business Name - Comp {i}'
            if comp_col in df.columns:
                df[f'Competitor_{i}_Name'] = df[comp_col]
                df[f'Competitor_{i}_Category'] = df.get(f'Category - Primary.{i}', 'Unknown')

        # Clean up and standardize
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.reset_index(drop=True, inplace=True)
        
        # Fill missing values strategically
        fill_values = {
            'Primary Category': 'Unknown',
            'State': 'Unknown', 
            'City': 'Unknown',
            'Sales Rep Name': 'Unassigned'
        }
        
        for col, value in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        
        # Fill remaining with 'Not Available'
        df.fillna('Not Available', inplace=True)

        print(f"Data processing complete. Total prospects: {len(df)}")
        print(f"Key columns: {[col for col in df.columns if col in ['Prospect Business Name', 'Primary Category', 'State', 'SEM', 'Google Places']]}")
        
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return pd.DataFrame()