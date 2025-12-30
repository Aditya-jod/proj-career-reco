import pandas as pd
 
class UniversityRecommender:
    def __init__(self, indian_df, world_df):
        self.indian_df = indian_df
        self.world_df = world_df

        self.field_keywords = {
            'STEM': ['Technology', 'Engineering', 'Science', 'Polytechnic', 'Technical', 'Institute of Technology'],
            'Healthcare': ['Medical', 'Health', 'Nursing', 'Pharmacy', 'Medicine', 'Hospital'],
            'Business_Finance': ['Business', 'Management', 'Commerce', 'Economics', 'Finance'],
            'Arts_Media': ['Arts', 'Design', 'Media', 'Music', 'Fine Arts', 'Communication'],
            'Government_Law': ['Law', 'Legal', 'Political', 'Public', 'Administration'],
            'Education': ['Education', 'Teacher', 'Pedagogy', 'Normal'],
            'Trades_Manufacturing': ['Industrial', 'Vocational', 'Manufacturing', 'Trade'],
            'Social_Services': ['Social', 'Welfare', 'Humanities', 'Community']
        }


    def recommend(self, career_field, country_preference='India'):
        keywords = self.field_keywords.get(career_field, [])
        if not keywords:
            return pd.DataFrame()
        results = pd.DataFrame()

        # Normalize location input
        location = country_preference.strip().lower()

        # For India, allow filtering by state or city/district as well
        if location == 'india' or location in self.indian_df['State'].str.lower().unique() or location in self.indian_df['District'].str.lower().unique():
            pattern = '|'.join(keywords)
            mask = self.indian_df['College Name'].str.contains(pattern, case=False, na=False)
            matches = self.indian_df[mask]

            # Further filter by state or district if location is not 'india'
            if location != 'india':
                state_mask = self.indian_df['State'].str.lower() == location
                district_mask = self.indian_df['District'].str.lower() == location
                matches = matches[state_mask | district_mask]

            # Prefer ranking if available, else prioritize top institutes, else fallback to random
            if 'Rank' in matches.columns and not matches.empty:
                results = matches.sort_values('Rank').head(10)[['College Name', 'State', 'District', 'Rank']]
            else:
                priority_keywords = ['IIT', 'NIT', 'IIM', 'BITS', 'AIIMS']
                priority_mask = matches['College Name'].str.contains('|'.join(priority_keywords), case=False, na=False)
                priority_matches = matches[priority_mask]
                other_matches = matches[~priority_mask]
                results = pd.concat([priority_matches, other_matches])
                if len(results) > 10:
                    results = results.head(10)[['College Name', 'State', 'District']]
                else:
                    results = results[['College Name', 'State', 'District']]
            if not results.empty:
                results.columns = ['University/College', 'State', 'City/District']
            return results

        else:
            country_map = {
                'usa': 'United States',
                'us': 'United States',
                'uk': 'United Kingdom',
                'uae': 'United Arab Emirates'
                ''
            }
            country_preference = country_map.get(country_preference.lower(), country_preference)

            country_mask = self.world_df['country'].str.contains(country_preference, case=False, na=False)
            country_df = self.world_df[country_mask]
            
            if country_df.empty:
                return pd.DataFrame()

            pattern = '|'.join(keywords)
            name_mask = country_df['name'].str.contains(pattern, case=False, na=False)
            results = country_df[name_mask][['name', 'country', 'web_pages']].head(10)
            
            if len(results) < 3:
                print(f"   (Note: Few specific matches found for '{career_field}'. Adding general top universities.)")
                general_unis = country_df[['name', 'country', 'web_pages']].head(5)
                results = pd.concat([results, general_unis]).drop_duplicates(subset=['name']).head(10)
            
            results.columns = ['University/College', 'Country', 'Website']
            
        return results