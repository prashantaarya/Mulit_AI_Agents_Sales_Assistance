# tools/hybrid_search.py
import pandas as pd
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

@dataclass
class SearchCriteria:
    categories: List[str] = None
    locations: Dict[str, str] = None
    digital_requirements: Dict[str, str] = None
    business_context: Dict[str, str] = None
    exclude_criteria: Dict[str, Any] = None
    sort_preferences: List[str] = None
    limit: int = 50

class HybridSearchToolBox:
    def __init__(self, df, llm):
        self.df = df.copy()
        self.llm = llm
        
        # Update column mapping to match your actual data structure
        self.column_map = {
            'customer': 'Customer',
            'products': 'Products', 
            'business_name': 'Prospect Business Name',  # Updated
            'category_primary': 'Primary Category',     # Updated  
            'category_secondary': 'Category - Secondary',
            'state': 'State',
            'city': 'City',
            'phone': 'Phone',
            'email': 'Email',
            'website': 'Website URL',
            'sales_rep': 'User Name',
            'facebook': 'FB latest_posts',
            'instagram': 'Insta Latest Posts',
            'twitter': 'Latest Tweets',
            'reviews': 'Reviews (local and social)',
            'google_ads': 'SEM',
            'google_places': 'Google Places',           # Added
            'display_ads': 'Display ads',
            'comp1_name': 'Business Name - Comp 1',
            'comp2_name': 'Business Name - Comp 2',
            'comp3_name': 'Business Name - Comp 3'
        }
        
        try:
            self.dataset_schema = self._analyze_dataset_schema()
            self.sample_data = self._get_sample_data()
            
            print(f"Dataset loaded: {len(df)} rows")
            self._analyze_data_distribution()
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            # Fallback minimal initialization
            self.dataset_schema = {'columns': list(df.columns), 'sample_values': {}}
            self.sample_data = []

    def _analyze_dataset_schema(self) -> Dict[str, Any]:
        """Analyze dataset structure for LLM understanding"""
        try:
            schema = {
                'columns': list(self.df.columns),
                'data_types': {},
                'unique_counts': {},
                'sample_values': {},
                'null_counts': {}
            }
            
            # Safe data type extraction
            for col in self.df.columns:
                try:
                    schema['data_types'][col] = str(self.df[col].dtype)
                    schema['unique_counts'][col] = int(self.df[col].nunique())
                    schema['null_counts'][col] = int(self.df[col].isnull().sum())
                except Exception as e:
                    print(f"Error analyzing column {col}: {e}")
                    schema['data_types'][col] = 'unknown'
                    schema['unique_counts'][col] = 0
                    schema['null_counts'][col] = 0
            
            # Get sample values for categorical columns
            for col in self.df.columns:
                try:
                    if self.df[col].dtype == 'object' and self.df[col].nunique() < 100:
                        unique_vals = self.df[col].dropna().unique()[:10]
                        # Convert to strings to avoid unhashable types
                        schema['sample_values'][col] = [str(val) for val in unique_vals if pd.notna(val)]
                except Exception as e:
                    print(f"Error getting sample values for {col}: {e}")
                    schema['sample_values'][col] = []
            
            return schema
            
        except Exception as e:
            print(f"Error in schema analysis: {e}")
            return {
                'columns': list(self.df.columns) if hasattr(self.df, 'columns') else [],
                'data_types': {},
                'unique_counts': {},
                'sample_values': {},
                'null_counts': {}
            }
    
    def _get_sample_data(self, n_samples: int = 3) -> List[Dict]:
        """Get sample records for LLM context"""
        try:
            sample_df = self.df.head(n_samples)
            # Convert to records and handle unhashable types
            records = []
            for _, row in sample_df.iterrows():
                record = {}
                for col in sample_df.columns:
                    value = row[col]
                    # Convert unhashable types to strings
                    if isinstance(value, (dict, list, set)):
                        record[col] = str(value)
                    elif pd.isna(value):
                        record[col] = 'N/A'
                    else:
                        record[col] = str(value)
                records.append(record)
            return records
        except Exception as e:
            print(f"Error in sample data generation: {e}")
            return []

    def find_prospects_hybrid(self, query: str) -> Dict:
        """Main search function using LLM intelligence"""
        print(f"\n=== PROCESSING: {query} ===")
        
        try:
            # Step 1: LLM analyzes the query and creates search criteria
            search_criteria = self._llm_analyze_query(query)
            
            # Step 2: Execute search based on LLM-generated criteria
            df_result = self._execute_intelligent_search(search_criteria)
            
            # Step 3: Format results
            if len(df_result) > 0:
                results = self._format_results(df_result.head(search_criteria.limit))
                return {
                    "prospects": results,
                    "total_found": len(df_result),
                    "showing": len(results),
                    "filters_applied": self._get_applied_filters(search_criteria),
                    "query": query
                }
            else:
                return self._provide_smart_suggestions(query, [])
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return {"error": f"Search failed: {str(e)}", "prospects": []}

    def _llm_analyze_query(self, query: str) -> SearchCriteria:
        """Use LLM to understand query and generate search criteria"""
        
        prompt = f"""
You are analyzing a business prospect database query. Generate precise search criteria.

DATASET SCHEMA:
Columns: {self.dataset_schema['columns']}
Sample values: {json.dumps(self.dataset_schema['sample_values'], indent=2)}
Total records: {len(self.df)}

USER QUERY: "{query}"

Generate JSON response with search criteria:

{{
    "categories": ["exact category names from Category - Primary that match"],
    "locations": {{"state": "state code", "city": "city name"}},
    "digital_requirements": {{"column_name": "Yes/No for digital presence"}},
    "business_context": {{"customer": "customer name", "sales_rep": "rep name"}},
    "exclude_criteria": {{"column_name": "values to exclude"}},
    "sort_preferences": ["preference1", "preference2"],
    "limit": 50,
    "reasoning": "Brief explanation"
}}

Rules:
1. Use exact column names from schema
2. For categories, match from sample values when possible
3. Map digital requirements to actual columns (FB latest_posts, SEM, etc.)
4. Extract state codes and city names for locations
5. Set appropriate limit based on query context

Generate ONLY the JSON response:
"""

        try:
            llm_response = self._call_llm(prompt)
            response_data = json.loads(llm_response.strip())
            
            criteria = SearchCriteria(
                categories=response_data.get('categories'),
                locations=response_data.get('locations'),
                digital_requirements=response_data.get('digital_requirements'),
                business_context=response_data.get('business_context'),
                exclude_criteria=response_data.get('exclude_criteria'),
                sort_preferences=response_data.get('sort_preferences', []),
                limit=response_data.get('limit', 50)
            )
            
            print(f"LLM Reasoning: {response_data.get('reasoning', 'No reasoning provided')}")
            return criteria
            
        except json.JSONDecodeError as e:
            print(f"LLM response parsing error: {e}")
            return SearchCriteria(limit=50)
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return SearchCriteria(limit=50)

    def _execute_intelligent_search(self, criteria: SearchCriteria) -> pd.DataFrame:
        """Execute search based on LLM-generated criteria"""
        df = self.df.copy()
        
        # Apply category filters
        if criteria.categories:
            category_col = self.column_map['category_primary']
            if category_col in df.columns:
                category_mask = df[category_col].isin(criteria.categories)
                df = df[category_mask]
                print(f"After category filter: {len(df)} records")
        
        # Apply location filters
        if criteria.locations:
            for loc_type, value in criteria.locations.items():
                col_name = self.column_map.get(loc_type)
                if col_name and col_name in df.columns:
                    if loc_type == 'state':
                        df = df[df[col_name].str.upper() == value.upper()]
                    elif loc_type == 'city':
                        df = df[df[col_name].str.contains(value, case=False, na=False)]
            print(f"After location filter: {len(df)} records")
        
        # Apply digital requirements
        if criteria.digital_requirements:
            for digital_col, requirement in criteria.digital_requirements.items():
                # Map to actual column names
                actual_col = None
                for key, col_name in self.column_map.items():
                    if digital_col.lower() in key.lower() or digital_col in col_name:
                        actual_col = col_name
                        break
                
                if actual_col and actual_col in df.columns:
                    if requirement.upper() == 'YES':
                        df = df[df[actual_col].notna() & (df[actual_col].str.upper() != 'NO')]
                    elif requirement.upper() == 'NO':
                        df = df[df[actual_col].isna() | (df[actual_col].str.upper() == 'NO')]
            print(f"After digital filter: {len(df)} records")
        
        # Apply business context filters
        if criteria.business_context:
            for context_type, value in criteria.business_context.items():
                col_name = self.column_map.get(context_type)
                if col_name and col_name in df.columns:
                    df = df[df[col_name].str.contains(value, case=False, na=False)]
            print(f"After business context filter: {len(df)} records")
        
        # Apply exclusion criteria
        if criteria.exclude_criteria:
            for col, exclude_values in criteria.exclude_criteria.items():
                if col in df.columns:
                    if isinstance(exclude_values, list):
                        df = df[~df[col].isin(exclude_values)]
                    else:
                        df = df[df[col] != exclude_values]
        
        return df

    def _get_applied_filters(self, criteria: SearchCriteria) -> List[str]:
        """Get list of applied filters for response"""
        filters = []
        
        if criteria.categories:
            filters.extend([f"Category: {cat}" for cat in criteria.categories])
        
        if criteria.locations:
            filters.extend([f"Location {k}: {v}" for k, v in criteria.locations.items()])
        
        if criteria.digital_requirements:
            filters.extend([f"Digital {k}: {v}" for k, v in criteria.digital_requirements.items()])
        
        if criteria.business_context:
            filters.extend([f"Business {k}: {v}" for k, v in criteria.business_context.items()])
        
        return filters

    def _format_results(self, df: pd.DataFrame) -> List[Dict]:
        """Format results for display - keep original structure"""
        results = []
        
        for _, row in df.iterrows():
            # Basic info
            result = {
                'business_name': str(row.get(self.column_map['business_name'], 'Unknown')),
                'category': str(row.get(self.column_map['category_primary'], 'Unknown')),
                'location': f"{row.get(self.column_map['city'], '')}, {row.get(self.column_map['state'], '')}".strip(', '),
            }
            
            # Digital presence summary
            digital_signals = []
            if row.get(self.column_map['google_ads']) == 'Yes':
                digital_signals.append('Google Ads')
            if row.get(self.column_map['facebook']) == 'Yes':
                digital_signals.append('Facebook')
            if row.get(self.column_map['reviews']) == 'Yes':
                digital_signals.append('Reviews')
                
            result['digital_presence'] = ', '.join(digital_signals) if digital_signals else 'Limited'
            
            # Contact info
            result['contact'] = {
                'phone': str(row.get(self.column_map['phone'], 'N/A')),
                'email': str(row.get(self.column_map['email'], 'N/A')),
                'website': str(row.get(self.column_map['website'], 'N/A'))
            }
            
            # Business context
            result['customer'] = str(row.get(self.column_map['customer'], 'Unknown'))
            result['sales_rep'] = str(row.get(self.column_map['sales_rep'], 'Unassigned'))
            
            results.append(result)
        
        return results

    def _provide_smart_suggestions(self, query: str, filters_applied: List[str]) -> Dict:
        """Provide smart suggestions using LLM when no results found"""
        
        availability_summary = {
            'total_records': len(self.df),
            'available_categories': {},
            'available_states': {},
            'digital_signal_stats': {}
        }
        
        # Get actual data availability
        cat_col = self.column_map.get('category_primary')
        if cat_col and cat_col in self.df.columns:
            availability_summary['available_categories'] = self.df[cat_col].value_counts().head(10).to_dict()
        
        state_col = self.column_map.get('state')
        if state_col and state_col in self.df.columns:
            availability_summary['available_states'] = self.df[state_col].value_counts().head(10).to_dict()
        
        # Digital signal stats
        for signal_key, col_name in [('SEM', self.column_map['google_ads']), 
                                    ('Reviews', self.column_map['reviews']), 
                                    ('Facebook', self.column_map['facebook'])]:
            if col_name in self.df.columns:
                availability_summary['digital_signal_stats'][signal_key] = (self.df[col_name] == 'Yes').sum()
        
        suggestions_prompt = f"""
User query "{query}" returned no results.

AVAILABLE DATA:
{json.dumps(availability_summary, indent=2)}

Generate helpful suggestions in JSON format:
{{
    "message": "Explanation of why no results found",
    "suggestions": [
        "Alternative search suggestion 1",
        "Alternative search suggestion 2"
    ],
    "data_insights": [
        "Available data insight 1",
        "Available data insight 2"
    ]
}}
"""
        
        try:
            response = self._call_llm(suggestions_prompt)
            suggestion_data = json.loads(response.strip())
            
            return {
                "prospects": [],
                "total_found": 0,
                "query": query,
                "filters_applied": filters_applied,
                "suggestions": suggestion_data.get('suggestions', []),
                "message": suggestion_data.get('message', 'No results found.')
            }
        except:
            return {
                "prospects": [],
                "total_found": 0,
                "query": query,
                "filters_applied": filters_applied,
                "suggestions": ["Try broader search terms", "Check category names", "Verify location spelling"],
                "message": "No results found. Try adjusting your search criteria."
            }

    def get_prospect_details(self, prospect_name: str) -> Optional[Dict]:
        """Get optimized prospect analysis for insights agent"""
        try:
            business_col = self.column_map['business_name']
            
            match = self.df[self.df[business_col].str.contains(prospect_name, case=False, na=False)]
            if match.empty:
                return {"error": f"No business found containing '{prospect_name}'"}
            
            prospect = match.iloc[0]
            
            # Calculate digital score efficiently
            digital_score = 0
            signals = ['google_ads', 'facebook', 'instagram', 'twitter', 'reviews', 'display_ads']
            active_signals = []
            missing_signals = []
            
            for signal in signals:
                col_name = self.column_map.get(signal)
                if col_name and col_name in self.df.columns:
                    value = prospect.get(col_name, 'No')
                    if str(value).upper() == 'YES':
                        digital_score += 1
                        active_signals.append(signal.replace('_', ' ').title())
                    else:
                        missing_signals.append(signal.replace('_', ' ').title())
            
            # Count competitors
            competitor_count = 0
            for i in range(1, 4):
                comp_col = self.column_map[f'comp{i}_name']
                if comp_col in self.df.columns:
                    comp_name = prospect.get(comp_col)
                    if pd.notna(comp_name) and str(comp_name).strip():
                        competitor_count += 1
            
            # Return compact analysis
            return {
                'business_name': str(prospect.get(business_col)),
                'category': str(prospect.get(self.column_map['category_primary'], 'Unknown')),
                'location': f"{prospect.get(self.column_map['city'], '')}, {prospect.get(self.column_map['state'], '')}".strip(', '),
                'digital_score': f"{digital_score}/6",
                'active_channels': ', '.join(active_signals[:3]) if active_signals else 'None',
                'missing_channels': ', '.join(missing_signals[:3]) if missing_signals else 'None',
                'competitor_count': competitor_count,
                'sales_rep': str(prospect.get(self.column_map['sales_rep'], 'Unassigned')),
                'top_opportunities': missing_signals[:2] if missing_signals else ['Digital optimization']
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
        
    def _analyze_data_distribution(self):
        """Analyze the actual data to understand patterns - keep original"""
        try:
            print("\n=== DATASET ANALYSIS ===")
            
            # Categories analysis - using actual column name
            category_col = self.column_map.get('category_primary')
            if category_col and category_col in self.df.columns:
                try:
                    categories = self.df[category_col].value_counts().head(10)
                    print(f"Top Categories: {dict(categories)}")
                except Exception as e:
                    print(f"Error analyzing categories: {e}")
            
            # States analysis
            state_col = self.column_map.get('state')
            if state_col and state_col in self.df.columns:
                try:
                    states = self.df[state_col].value_counts().head(10)
                    print(f"Top States: {dict(states)}")
                except Exception as e:
                    print(f"Error analyzing states: {e}")
            
            # Digital signals analysis
            digital_cols = ['google_ads', 'facebook', 'reviews', 'display_ads', 'google_places']
            for signal in digital_cols:
                col_name = self.column_map.get(signal)
                if col_name and col_name in self.df.columns:
                    try:
                        values = self.df[col_name].value_counts()
                        print(f"{signal.title()}: {dict(values)}")
                    except Exception as e:
                        print(f"Error analyzing {signal}: {e}")
                        
        except Exception as e:
            print(f"Error in data distribution analysis: {e}")

    def _call_llm(self, prompt: str) -> str:
        """Call LLM service - customize based on your provider"""
        try:
            # For OpenAI
            if hasattr(self.llm, 'chat'):
                response = self.llm.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            # For Anthropic Claude
            elif hasattr(self.llm, 'messages'):
                response = self.llm.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            # Fallback mock for testing
            else:
                return self._mock_llm_response(prompt)
                
        except Exception as e:
            print(f"LLM call failed: {e}")
            return self._mock_llm_response(prompt)
    
    def _mock_llm_response(self, prompt: str) -> str:
        """Mock LLM response for testing"""
        if "generate search criteria" in prompt.lower():
            return json.dumps({
                "categories": [],
                "locations": {},
                "digital_requirements": {},
                "business_context": {},
                "sort_preferences": [],
                "limit": 50,
                "reasoning": "Mock response for testing"
            })
        else:
            return json.dumps({
                "message": "No results found for your query.",
                "suggestions": ["Try different search terms", "Check spelling"],
                "data_insights": ["Dataset contains multiple categories"]
            })