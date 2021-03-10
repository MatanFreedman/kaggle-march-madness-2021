# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd
import requests

def scrape_kp_season(season):
    """Scrapes kenpom data for a season from kenpom.com
    
    Parameters
    ----------
    season : int/str
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Season {season}")

    # request data:
    url = f"http://www.kenpom.com/index.php?y={season}"
    html = requests.get(url)
    soup = BeautifulSoup(html.content, features="html.parser")
    
    # find table and parse:
    table = soup.find("table", attrs= {'id':'ratings-table'})
    
    # 2 rows of headers that repeat throughout table:
    headings1 = [th.get_text() for th in table.find("tr", attrs={"class":"thead1"}).find_all("th")]
    headings2 = [th.get_text() for th in table.find("tr", attrs={"class":"thead2"}).find_all("th")]
    headings = [(h1 + "_" + h2).lstrip("_") for h1, h2 in zip(headings1, headings2)]

    # initialize dataframe, loop and append
    kp_data = pd.DataFrame(columns=headings)
    tables = table.find_all("tbody")
    for table in tables:
        for row in table.find_all("tr"):
            dataset = list(zip(headings, (td.get_text() for td in row.find_all("td"))))
            kp_data = kp_data.append(dict(dataset), ignore_index=True)
    
    # still caught some empty rows:
    kp_data = kp_data.dropna(how='all').reset_index(drop=True)
    
    kp_data['Season'] = season
    return kp_data


def kp_data():
    """Scrapes KP data from 2003-2021
    """
    logger = logging.getLogger(__name__)
    logger.info("Scraping Pomeroy Basketball Ratings.")

    # initialize kp data and loop thru seasons.
    kenpom_df = pd.DataFrame()
    for season in range(2003, 2022):
        logger.debug(f"Season: {season}")
        df = scrape_kp_season(season)
        kenpom_df = kenpom_df.append(df)

    return kenpom_df
    


def main():
    logger = logging.getLogger(__name__)

    proj_path = Path().resolve().parents[1]
    raw_data_path = proj_path / "data" / "raw" 

    kenpom_df = kp_data()
    kp_path = raw_data_path / "kenpom.csv"
    kenpom_df.to_csv(kp_path, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
