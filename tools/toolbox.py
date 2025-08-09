# tools/toolbox.py
from typing import Dict, List, Optional
from langchain.tools import tool

class ToolBox:
    """Container for simple, single-purpose data access tools."""
    def __init__(self, dataframe):
        self.df = dataframe
        # Simple check for required columns to avoid errors later
        if 'Prospect Business Name' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Prospect Business Name' column.")
        if 'Primary Category' not in self.df.columns:
            print("⚠️ Warning: 'Primary Category' column not found. Category search will be disabled.")

    @tool
    def search_prospects(self, category: Optional[str] = None, location: Optional[str] = None, has_google_ads: Optional[bool] = None) -> List[Dict]:
        """
        Searches for prospects based on structured criteria.

        Args:
            category (Optional[str]): The primary business category (e.g., "Computer Contractors").
            location (Optional[str]): The state of the prospect (e.g., "Texas").
            has_google_ads (Optional[bool]): Whether the prospect is known to use Google Ads.

        Returns:
            List[Dict]: A list of prospect data dictionaries matching the criteria.
        """
        filtered_df = self.df.copy()

        if category and 'Primary Category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Primary Category'].str.contains(category, case=False, na=False)]

        if location and 'State' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['State'].str.contains(location, case=False, na=False)]

        if has_google_ads is not None and 'BuzzBoard Data Parsed' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['BuzzBoard Data Parsed'].apply(
                    lambda x: (x.get('Advertising', {}).get('Google Ads', 'No') == 'Yes') == has_google_ads
                )
            ]

        # Return a limited set of key information for brevity
        results = filtered_df[['Prospect Business Name', 'Primary Category', 'City', 'State']].head(10).to_dict('records')
        return results

    @tool
    def get_prospect_details(self, prospect_name: str) -> Optional[Dict]:
        """
        Retrieves detailed information, including parsed BuzzBoard data, for a single prospect by name.

        Args:
            prospect_name (str): The exact business name of the prospect.

        Returns:
            Optional[Dict]: A dictionary containing the prospect's detailed data, or None if not found.
        """
        # Find the prospect by name (case-insensitive)
        match = self.df[self.df['Prospect Business Name'].str.lower() == prospect_name.lower()]

        if not match.empty:
            # Return all details for the found prospect
            return match.iloc[0].to_dict()

        return None