import pandas as pd
 
class UniversityRecomender:
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
            results = self.indian_df[mask][['College Name', 'State', 'District']].head(10)
            results.columns = ['University/College', 'State', 'City/District']

        else:
            country_mask = self.world_df['country'].str.contains(country_preference, case=False, na=False)
            country_df = self.world_df[country_mask]

            pattern = '|'.join(keywords)
            name_mask = country_df['name'].str.contains(pattern, case=False, na=False)

            results = country_df[name_mask][['name', 'country', 'web_pages']].head(10)
            results.columns = ['University/College', 'Country', 'Website']
            
        return results