import pandas as pd

def load_augmentation(number_of_samples = 9000):
    """Load Amazon fine food review data, format it to augement the original food review data"""
    
    df = pd.read_csv('./amazon_food/Reviews.csv',usecols=['Score','Text'])
    df.columns = df.columns.str.lower()
    
    # Set score 1&2 to be negative, score 4&5 to be positive, score 3 is dropped
    df = df[df['score']!=3]
    df['label'] = 0
    df.loc[ df['score'].isin([4,5]), 'label'] = 1
    df = df[['label', 'text']]

    # Select the head
    df = pd.concat([df[df['label']==1].sample(int(9000/2), random_state = 42), df[df['label']==0].sample(int(9000/2), random_state = 42)]).reset_index(drop=True)
    
    return df