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

        if country_preference.lower() == 'india':
            pattern = '|'.join(keywords)

            mask = self.indian_df['College Name'].str.contains(pattern, case=False, na=False)
            matches = self.indian_df[mask]
            
            # Randomize results to show diversity across states
            if len(matches) > 10:
                results = matches.sample(n=10, random_state=None)[['College Name', 'State', 'District']]
            else:
                results = matches[['College Name', 'State', 'District']]
                
            results.columns = ['University/College', 'State', 'City/District']

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