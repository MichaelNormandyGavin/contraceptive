import requests
import re
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data'

pattern = '\d+'

columns = ['wife_age','wife_edu','husb_edu','num_child','wife_islam','wife_unemployed','husb_occupation','life_std','bad_media','contra_type']

def make_text(url=url,pattern=pattern,as_int=True):
    '''function to read the file from UCI's machine learning repository:
        
        url = string; where the data is hosted
        pattern = string; for re.findall search
        astype = Boolean; return list values as integers for data analysis'''
    
    file = requests.get(url)
    
    rows = file.text.split(sep='\n')
    
    if as_int:
        text_file = [list(map(int,re.findall(pattern,l))) for l in rows]
    
    else:
        text_file = [re.findall(pattern,l) for l in rows]
        
    return text_file

def make_df(columns=columns,**kwargs):
    '''Build Pandas DataFrame from make_text object
    
    columns = list; column headers for DataFrame
    **kwargs = kwargs; used to feed into make_text function'''
    
    
    text_file = make_text(**kwargs)
    
    df = pd.DataFrame.from_records(text_file[:-1],columns=columns)
    
    return df