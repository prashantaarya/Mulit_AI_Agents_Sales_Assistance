# tools/hybrid_search.py
import pandas as pd
import ast
from typing import Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import Literal

class FilterCondition(BaseModel):
    field: str = Field(..., description="Column name to filter on")
    operator: Literal['contains', 'not_contains', 'equals', 'not_equals', 'greater_than', 'less_than'] = Field(..., description="Filter operator")
    value: str = Field(..., description="Value to search for")

class FilterList(BaseModel):
    filters: List[FilterCondition]

class HybridSearchToolBox:
    def __init__(self, df: pd.DataFrame, llm):
        self.df = df
        self.llm = llm
        self.column_descriptions = self._get_column_descriptions()
        self._create_content_vector_store()
        self._create_extractor_chain()

    def _get_column_descriptions(self) -> dict:
        """Get descriptions for available columns."""
        descriptions = {
            'Prospect Business Name': 'Business name of the prospect',
            'Primary Category': 'Main industry category',
            'City': 'Business location city',
            'State': 'Business location state', 
            'Customer': 'Customer/client name',
            'Sales Rep Name': 'Assigned sales representative',
            'Website': 'Business website URL'
        }
        
        # Add digital signal descriptions if columns exist
        digital_signals = {
            'SEM': 'Google Ads/Paid search activity (Yes/No)',
            'FB latest_posts': 'Facebook posting activity (Yes/No)', 
            'Reviews (local and social)': 'Review count and social presence',
            'Display ads': 'Display advertising activity (Yes/No)',
            'Google Places': 'Google My Business listing (Yes/No)'
        }
        
        for col, desc in digital_signals.items():
            if col in self.df.columns:
                descriptions[col] = desc
                
        # Add competitor descriptions
        for i in range(1, 4):
            comp_col = f'Competitor_{i}_Name'
            if comp_col in self.df.columns:
                descriptions[comp_col] = f'Competitor {i} name'
                
        return descriptions

    def _create_content_vector_store(self):
        """Create vector store for semantic search."""
        print("Creating content vector store...")
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            documents = []
            for index, row in self.df.iterrows():
                content_parts = []
                
                # Basic business info
                if pd.notna(row.get('Prospect Business Name')):
                    content_parts.append(f"Business: {row.get('Prospect Business Name')}")
                if pd.notna(row.get('Primary Category')):
                    content_parts.append(f"Category: {row.get('Primary Category')}")
                if pd.notna(row.get('City')) and pd.notna(row.get('State')):
                    content_parts.append(f"Location: {row.get('City')}, {row.get('State')}")
                
                # Digital signals
                digital_signals = []
                for signal in ['SEM', 'FB latest_posts', 'Google Places', 'Display ads']:
                    if signal in self.df.columns and row.get(signal) == 'Yes':
                        digital_signals.append(signal.replace('_', ' '))
                
                if digital_signals:
                    content_parts.append(f"Digital: {', '.join(digital_signals)}")
                
                content = '. '.join(content_parts) if content_parts else f"Business {index}"
                documents.append(Document(page_content=content, metadata={"index": index}))
            
            self.content_vector_store = FAISS.from_documents(documents, embedding_model)
            print("Content vector store created successfully.")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.content_vector_store = None

    def _create_extractor_chain(self):
        """Create LLM chain for extracting filters."""
        try:
            parser = self.llm.with_structured_output(FilterList)
            prompt = ChatPromptTemplate.from_template(
                "Extract filters from the user query.\n\n"
                "Available fields:\n{columns}\n\n"
                "QUERY INTERPRETATION RULES:\n"
                "- 'Computer Contractors' -> field='Primary Category', operator='contains', value='Computer'\n"
                "- 'in Texas' -> field='State', operator='equals', value='TX'\n"  
                "- 'with Google ads' -> field='SEM', operator='equals', value='Yes'\n"
                "- 'low local presence' -> field='Google Places', operator='equals', value='No'\n"
                "- 'high spend' typically refers to 'SEM'='Yes'\n\n"
                "Query: {query}"
            )
            self.extractor_chain = prompt | parser
        except Exception as e:
            print(f"Error creating extractor chain: {e}")
            self.extractor_chain = None

    def find_prospects_hybrid(self, query: str) -> List[Dict]:
        """Intelligent hybrid search with error handling."""
        print(f"Processing query: {query}")

        try:
            # Extract filters
            if self.extractor_chain:
                try:
                    column_info = "\n".join([f"- {name}: {desc}" for name, desc in self.column_descriptions.items()])
                    filter_obj = self.extractor_chain.invoke({"query": query, "columns": column_info})
                    filters = filter_obj.filters if hasattr(filter_obj, 'filters') else []
                except Exception as e:
                    print(f"Filter extraction error: {e}")
                    filters = self._parse_query_manually(query)
            else:
                filters = self._parse_query_manually(query)

            print(f"Extracted filters: {[(f.field, f.operator, f.value) for f in filters]}")
            
            # Apply filters
            filtered_df = self._apply_filters_safe(self.df.copy(), filters)
            print(f"Results after filtering: {len(filtered_df)}")

            if filtered_df.empty:
                return self._handle_empty_results(query)

            # Return structured results
            results = self._format_results(filtered_df, query)
            return results[:10]  # Limit to top 10
            
        except Exception as e:
            print(f"Search error: {e}")
            return [{"error": f"Search failed: {str(e)}", "suggestion": "Try a simpler query"}]

    def _parse_query_manually(self, query: str) -> List[FilterCondition]:
        """Manual query parsing as fallback."""
        filters = []
        query_lower = query.lower()
        
        # Category filters
        if 'computer' in query_lower:
            filters.append(FilterCondition(field='Primary Category', operator='contains', value='Computer'))
        
        # Location filters  
        if 'texas' in query_lower or ' tx ' in query_lower:
            filters.append(FilterCondition(field='State', operator='equals', value='TX'))
        if 'california' in query_lower or ' ca ' in query_lower:
            filters.append(FilterCondition(field='State', operator='equals', value='CA'))
            
        # Digital presence filters
        if 'google ads' in query_lower or 'high.*spend' in query_lower:
            filters.append(FilterCondition(field='SEM', operator='equals', value='Yes'))
        if 'low.*local' in query_lower or 'no.*local' in query_lower:
            filters.append(FilterCondition(field='Google Places', operator='equals', value='No'))
        if 'social media' in query_lower:
            filters.append(FilterCondition(field='FB latest_posts', operator='equals', value='Yes'))
            
        return filters

    def _apply_filters_safe(self, df: pd.DataFrame, filters: List[FilterCondition]) -> pd.DataFrame:
        """Apply filters with comprehensive error handling."""
        for f in filters:
            try:
                if f.field not in df.columns:
                    print(f"Column '{f.field}' not found. Available: {list(df.columns[:5])}")
                    continue

                col_data = df[f.field].astype(str)
                filter_value = str(f.value).strip()

                if f.operator == 'contains':
                    mask = col_data.str.contains(filter_value, case=False, na=False, regex=False)
                elif f.operator == 'equals':
                    if f.field == 'State' and len(filter_value) > 2:
                        filter_value = self._get_state_abbrev(filter_value)
                    mask = col_data.str.upper() == filter_value.upper()
                elif f.operator == 'not_equals':
                    mask = col_data.str.upper() != filter_value.upper()
                else:
                    continue  # Skip unsupported operators
                    
                df = df[mask]
                
            except Exception as e:
                print(f"Error applying filter {f.field}: {e}")
                continue

        return df

    def _format_results(self, df: pd.DataFrame, query: str) -> List[Dict]:
        """Format results in structured format."""
        results = []
        
        for _, row in df.iterrows():
            try:
                # Basic info
                business_name = str(row.get('Prospect Business Name', 'Unknown'))
                category = str(row.get('Primary Category', 'Unknown'))
                location = f"{str(row.get('City', 'Unknown'))}, {str(row.get('State', 'Unknown'))}"
                
                # Digital presence summary
                digital_status = []
                if row.get('SEM') == 'Yes':
                    digital_status.append('Google Ads Active')
                if row.get('Google Places') == 'Yes':
                    digital_status.append('Local SEO')
                else:
                    digital_status.append('Low Local Presence')
                if row.get('FB latest_posts') == 'Yes':
                    digital_status.append('Social Media')
                
                # Opportunities
                opportunities = []
                if row.get('Google Places') == 'No':
                    opportunities.append('Local SEO Setup')
                if row.get('SEM') == 'No':
                    opportunities.append('Google Ads')
                if row.get('FB latest_posts') == 'No':
                    opportunities.append('Social Media')
                
                result = {
                    'business_name': business_name,
                    'category': category,
                    'location': location,
                    'digital_status': ', '.join(digital_status) if digital_status else 'Limited Presence',
                    'opportunities': ', '.join(opportunities) if opportunities else 'Optimization',
                    'sales_rep': str(row.get('Sales Rep Name', 'Unassigned')),
                    'contact': {
                        'phone': str(row.get('Prospect Phone', 'Not Available')),
                        'email': str(row.get('Prospect Email', 'Not Available'))
                    }
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error formatting result: {e}")
                continue
                
        return results

    def _handle_empty_results(self, query: str) -> List[Dict]:
        """Handle when no results found."""
        # Show available categories/states as suggestions
        categories = self.df['Primary Category'].value_counts().head(5).to_dict()
        states = self.df['State'].value_counts().head(5).to_dict()
        
        return [{
            'message': 'No exact matches found',
            'suggestions': {
                'available_categories': categories,
                'available_states': states,
                'sample_query': 'Find computer businesses in Texas'
            }
        }]

    def _get_state_abbrev(self, state_name: str) -> str:
        """Convert state name to abbreviation."""
        state_map = {
            'texas': 'TX', 'california': 'CA', 'florida': 'FL', 'new york': 'NY',
            'illinois': 'IL', 'pennsylvania': 'PA', 'ohio': 'OH', 'georgia': 'GA'
        }
        return state_map.get(state_name.lower(), state_name.upper()[:2])

    def get_prospect_details(self, prospect_name: str) -> Optional[Dict]:
        """Get detailed prospect information."""
        try:
            match = self.df[self.df['Prospect Business Name'].str.contains(prospect_name, case=False, na=False)]
            if match.empty:
                return None
                
            prospect = match.iloc[0]
            
            # Calculate scores
            digital_score = self._calculate_digital_score(prospect)
            
            return {
                'business_name': str(prospect.get('Prospect Business Name')),
                'category': str(prospect.get('Primary Category')),
                'location': f"{str(prospect.get('City'))}, {str(prospect.get('State'))}",
                'contact': {
                    'phone': str(prospect.get('Prospect Phone', 'Not Available')),
                    'email': str(prospect.get('Prospect Email', 'Not Available')),
                    'website': str(prospect.get('Website', 'Not Available'))
                },
                'digital_score': f"{digital_score}/10",
                'current_presence': {
                    'google_ads': str(prospect.get('SEM', 'No')),
                    'local_seo': str(prospect.get('Google Places', 'No')),
                    'social_media': str(prospect.get('FB latest_posts', 'No'))
                },
                'sales_rep': str(prospect.get('Sales Rep Name', 'Unassigned')),
                'opportunities': self._get_opportunities(prospect)
            }
            
        except Exception as e:
            print(f"Error getting prospect details: {e}")
            return None

    def _calculate_digital_score(self, row) -> int:
        """Calculate digital presence score."""
        score = 0
        if row.get('SEM') == 'Yes':
            score += 3
        if row.get('Google Places') == 'Yes':
            score += 2
        if row.get('FB latest_posts') == 'Yes':
            score += 2
        if row.get('Display ads') == 'Yes':
            score += 2
        return min(score, 10)

    def _get_opportunities(self, row) -> List[str]:
        """Identify opportunities for prospect."""
        opportunities = []
        if row.get('SEM') == 'No':
            opportunities.append('Google Ads Setup')
        if row.get('Google Places') == 'No':
            opportunities.append('Local SEO Optimization')
        if row.get('FB latest_posts') == 'No':
            opportunities.append('Social Media Management')
        return opportunities