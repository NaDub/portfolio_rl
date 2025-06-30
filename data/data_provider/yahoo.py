import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

class Yahoo:
    pass

def fetch_decorator(func):
    """
    Decorator function that fetches the content from a URL and 
    parses it using the provided parsing function.
    """
    def wrapper(url: str):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return func(soup)
        else:
            print(f"Request failure for {url}: {response.status_code}")
            return None
    return wrapper

@fetch_decorator
def parse_yahoo(soup:BeautifulSoup, _df=True):
    """
    Function to get data on yahoo.
    """

    # Find title
    text = soup.find_all('h1', class_='yf-xxbei9')[0].text.strip()
    title = re.findall(r"\((.*?)\)", text)[0]

    # Find columns
    columns = ['Date', 'Open', 'High', 'Low', 'Close','Adj_Close',
       'Volume', 'Title']

    # Find all cells and create list of dict
    rows = soup.find_all('tr', class_='yf-1jecxey')
    data = []
    for row in rows[1:]:
        cells = row.find_all('td', class_='yf-1jecxey')
        values = [cell.text.strip() for cell in cells]
        values.append(title)

        if len(values) == len(columns):
            stock_entry = {}
            stock_entry.update(dict(zip(columns, values)))
            data.append(stock_entry)
    
    if _df == True: return pd.DataFrame(data)   
    return data

def treat_yahoo(df:pd.DataFrame):
    """
    Convert Yahoo Finance data types to appropriate formats.
    - Date -> datetime
    - Open, High, Low, Close, Adj_Close -> float (handle commas in numbers)
    - Volume -> int (remove commas)
    """
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='%b %d, %Y', errors='coerce')

    num_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

    for col in num_cols:
        df_temp[col] = df_temp[col].astype(str).str.replace(',', '', regex=True)
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    df_temp['Volume'] = df_temp['Volume'].astype('Int64')

    return df_temp


if __name__ == '__main__':
    tester = parse_yahoo('https://finance.yahoo.com/quote/^FCHI/history/?period1=1262303940&period2=1739895424')
    print(tester.head())
    tester = treat_yahoo(tester)
    print(tester.head())