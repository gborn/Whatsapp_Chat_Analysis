
import pandas as pd
import re
import csv
import fire
import warnings
warnings.filterwarnings("ignore")

def separator(msg):
    """
    Extract datetime, person name or number, and message text from msg
    """
    # remove whitespaces
    msg = msg.strip()

    # we define three groups: datetime, person name/phone number, message
    result = re.search(r'(\d{1,2}/\d{1,2}/\d{4},\s+\d{1,2}:\d{1,2})\s+-\s+([+0-9a-zA-Z\s]+):\s+(.*)', msg)

    # ignore lines that don't have above three groups, and return empty line
    if not hasattr(result, 'group'):
        return ''

    return result.groups()




def txt_to_csv(fn):
    """
    Read lines from text file "fn",
    create and save a csv file
    """
    with open(fn) as f:
         lines = map(separator, f.readlines())

    with open(f'{fn[:-4]}.csv', 'w') as wf:
        out = csv.writer(wf)
        out.writerow(['datetime', 'id', 'message'])
        for line in lines:
            if line:
                out.writerow(line)





def add_datepart(df, fieldname):
    """
    Adds date related features to dataframe df inplace
    df: dataframe
    fieldname: name of the date field in df
    """
    new_df = df.copy()
    field = df[fieldname]
    target_prefix = re.sub('[Dd]atetime$', '', fieldname)
    
    date_features = (
         'hour',
         'minute',
         'Year', 
         'Month', 
         'Week', 
         'Day', 
         'Dayofweek', 
         'Dayofyear', 
    )
    
    for name in date_features:
        new_df[target_prefix+name] = getattr(field.dt, name.lower())
        
    new_df[target_prefix+'Elapsed'] = (field - field.min()).dt.days
    new_df[target_prefix+'MonthName'] = field.dt.month_name()
    new_df[target_prefix+'DayName'] = field.dt.day_name()
    new_df.drop(fieldname, axis=1, inplace=True)

    return new_df




def preprocess(fn:str) -> pd.DataFrame:
    """
    Preprocess whatsapp text file
    """
    txt_to_csv(fn)
    chats_df = pd.read_csv(f'{fn[:-4]}.csv', parse_dates=['datetime'])
    chats = chats_df.set_index('datetime')

    chats_with_features = add_datepart(chats_df, 'datetime')

    return chats, chats_with_features

if __name__ == '__main__':
    import sys
    if sys.argv:
        chats_df = preprocess(sys.argv[1])
        print(chats_df.head(5))
