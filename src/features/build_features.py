from pathlib import Path

import pandas as pd
import numpy as np 

import logging

def prepare_data(df):
    """Prepares the regular and tournament datasets.
    
    Parameter
    ---------
    df : pandas DataFrame using the tournament/regular season CSV files
    """
    # copies:
    winners = df.copy()
    losers = df.copy()

    # reset WLoc column:
    winners.columns.values[6] = 'location'
    losers.columns.values[6] = 'location'

    # switch H/A in losers df:
    losers.loc[losers['location'] == 'H', 'location'] = 'A'
    losers.loc[losers['location'] == 'A', 'location'] = 'H'

    # replace W and L with appropriate prefix:
    winners.columns = winners.columns.str.replace("W", "T1_").str.replace("L", "T2_")
    losers.columns = losers.columns.str.replace("W", "T2_").str.replace("L", "T1_")

    # combine:
    output = pd.concat([winners, losers]).sort_index().reset_index(drop=True)
    
    # change location preprocessing:
    output.loc[output.location=='N','location'] = '0'
    output.loc[output.location=='H','location'] = '1'
    output.loc[output.location=='A','location'] = '-1'
    output.location = output.location.astype(int)
    
    # calc point diff:
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output

def calc_season_statistics(regular_data):
    """Calc season statistics using Kaggle data
    """
    boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
        'PointDiff']

    funcs = [np.mean]

    # team and opponent regular season stats:
    season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
    T1_season_stats = season_statistics.copy()
    T2_season_stats = season_statistics.copy()

    T1_season_stats.columns = [''.join(col).strip() for col in T1_season_stats.columns.values]
    T1_season_stats.columns = T1_season_stats.columns.str.replace("T2_", "T1_opponent_")

    T2_season_stats.columns = [''.join(col).strip() for col in T2_season_stats.columns.values]
    T2_season_stats.columns = ["T2_" + s.replace("T1_", "").replace("T2_", "opponent_") for s in T2_season_stats.columns]

    T1_season_stats.columns.values[0] = 'Season'
    T2_season_stats.columns.values[0] = 'Season'
    return T1_season_stats, T2_season_stats

def win_ratio_14_days(regular_data):
    """Calculates win ratio column from prior 14 days
    """
    # Calc prior 2 weeks win %
    last14_days_T1 = regular_data.loc[regular_data['DayNum']>118].reset_index(drop=True)
    last14_days_T1['win'] = np.where(last14_days_T1['PointDiff'] > 0, 1, 0)
    last14_days_T1 = last14_days_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

    last14_days_T2 = regular_data.loc[regular_data['DayNum']>118].reset_index(drop=True)
    last14_days_T2['win'] = np.where(last14_days_T2['PointDiff'] < 0, 1, 0)
    last14_days_T2 = last14_days_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')
    return last14_days_T1, last14_days_T2

def calc_seed_diff(seeds):
    """Calcs "SeedDiff column using seeds df"""
    seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

    seeds_T1 = seeds[['Season', 'seed', 'TeamID']].copy()
    seeds_T2 = seeds[['Season', 'seed', 'TeamID']].copy()

    seeds_T1.columns = ['Season', 'T1_seed', 'T1_TeamID']
    seeds_T2.columns = ['Season', 'T2_seed', 'T2_TeamID']

    return seeds_T1, seeds_T2

def clean_kp_data(kp_data_raw, spellings, teams):
    """Cleans the kenpom data
    """
    logger = logging.getLogger(__name__)
    to_replace = {
        "\s?[0-9]":"",
        "(\s{1}st\.?$)": " state",
        "-": " ",
        "\(": "",
        "\)": "",
        "\**": "",
        "ut rio grande valley": "texas rio grande valley",
        "texas a&m corpus chris": "a&m corpus chris",
        "southwest missouri state":"sw missouri state",
        "texas a&m corpus christi": "a&m corpus christi",
        "cal st. bakersfield": "cal state bakersfield",
        "st. francis pa":"st francis pa",
        "troy state": "troy",
    }
    kenpom_df = kp_data_raw.copy()
    kenpom_df['TeamName'] = kenpom_df['Team'].str.lower()
    kenpom_df['TeamName'] = kenpom_df['TeamName'].replace(regex=to_replace)

    spellings['TeamName'] = spellings['TeamNameSpelling']
    all_team_names = teams[['TeamName', 'TeamID']].append(spellings[['TeamName', 'TeamID']])
    all_team_names['TeamName'] = all_team_names['TeamName'].str.lower()
    all_team_names['TeamName'] = all_team_names['TeamName'].str.replace("-", " ")
    all_team_names = all_team_names.drop_duplicates()

    kenpom_df = pd.merge(kenpom_df, all_team_names, on='TeamName', how='left')
    logger.debug(f"{kenpom_df['TeamID'].isna().sum()} missing ID's")

    kp_cols = ['Season', 'TeamID',
        'Rk', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck',
       'Strength of Schedule_AdjEM', 'Strength of Schedule_OppO',
       'Strength of Schedule_OppD', 'NCSOS_AdjEM'  
    ]

    kp_data = kenpom_df[kp_cols]
    T1_kp = kp_data.copy(deep=True)
    T2_kp = kp_data.copy(deep=True).reset_index(drop=True)

    T1_kp.columns = ["T1_" + c for c in T1_kp.columns]
    T1_kp.columns.values[0] = "Season"

    T2_kp.columns = ["T2_" + c for c in T2_kp.columns]
    T2_kp.columns.values[0] = 'Season'
    return T1_kp, T2_kp


def main():
    logger = logging.getLogger(__name__)
    logger.info("Building features")

    # use kenpom and kaggle data to create dataset
    proj_dir = Path().resolve()
    data_dir = proj_dir / "data" 

    logger.debug(proj_dir)

    # load kaggle data:
    tourney_results = pd.read_csv(data_dir / "external" / "MNCAATourneyDetailedResults.csv")
    seeds = pd.read_csv(data_dir / "external" / 'MNCAATourneySeeds.csv')
    regular_results = pd.read_csv(data_dir / "external" / 'MRegularSeasonDetailedResults.csv')
    teams = pd.read_csv(data_dir / "external" / "MTeams.csv")
    spellings = pd.read_csv(data_dir / "external" / "MTeamSpellings.csv", encoding = "ISO-8859-1")

    # load kenpom exteral data:
    kp_path = proj_dir / "data" / "raw" / "kenpom.csv"
    kp_data_raw = pd.read_csv(kp_path)

    # clean kp data:
    T1_kp, T2_kp = clean_kp_data(kp_data_raw, spellings, teams)

    #  prepare tourney and regular season data:
    regular_data = prepare_data(regular_results)
    tourney_data = prepare_data(tourney_results)

    # feature eng:
    # season statistics:
    T1_season_stats, T2_season_stats = calc_season_statistics(regular_data)
    last14_days_T1, last14_days_T2 = win_ratio_14_days(regular_data)
    seeds_T1, seeds_T2 = calc_seed_diff(seeds)

    # combine:
    tourney_data = pd.merge(tourney_data, T1_season_stats, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, T2_season_stats, on=['Season', 'T2_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, last14_days_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, last14_days_T2, on=['Season', 'T2_TeamID'], how='left')
    tourney_data = tourney_data.merge(T1_kp, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = tourney_data.merge(T2_kp, on=['Season', 'T2_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, seeds_T2, on=['Season', 'T2_TeamID'], how='left')
    tourney_data['SeedDiff'] = tourney_data['T1_seed'] - tourney_data['T2_seed']

    # save
    logger.info("Saving tourney_data to processed folder")
    tourney_data.to_csv(data_dir / "processed" / "tourney_data.csv", index=False)
    
if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    main()

