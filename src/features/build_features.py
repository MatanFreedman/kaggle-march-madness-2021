from pathlib import Path

import pandas as pd
import numpy as np 

import logging

logger = logging.getLogger(__name__)
logger.info("Building features")

def prepare_data(df):
    """Prepares the regular and tournament datasets.
    
    Parameter
    ---------
    df : pandas DataFrame using the tournament/regular season CSV files
    """
    # calc advanced stats:
    df = calc_advanced_stats(df)

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
    output['T1_PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output

def calc_advanced_stats(data):
    df = data.copy()
    logger.info("Calculating advanced stats")
    # Points Winning/Losing Team
    logger.debug("W/LPts")
    df['WPts'] = 2*df['WFGM'] + df['WFGM3'] + df['WFTM']
    df['LPts'] = 2 * df['LFGM'] + df['LFGM3'] + df['LFTM']

    #Calculate Winning/losing Team Possesion Feature
    logger.debug("Pos")
    wPos = 0.96*(df['WFGA'] + df['WTO'] + 0.44*df['WFTA'] - df['WOR'])
    lPos = 0.96*(df.LFGA + df.LTO + 0.44*df.LFTA - df.LOR)
    #two teams use almost the same number of possessions in a game
    #(plus/minus one or two - depending on how quarters end)
    #so let's just take the average
    df['Pos'] = (wPos+lPos)/2

    #Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
    logger.debug("W/L Offensive ratings")
    df['WOffRtg'] = 100 * (df.WPts / df.Pos)
    df['LOffRtg'] = 100 * (df.LPts / df.Pos)
    #Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
    logger.debug("Defensive ratings")
    df['WDefRtg'] = df.LOffRtg
    df['LDefRtg'] = df.WOffRtg
    #Net Rating = Off.Rtg - Def.Rtg
    df['WNetRtg'] = df.WOffRtg - df.WDefRtg
    df['LNetRtg'] = df.LOffRtg - df.LDefRtg
                         
    #Assist Ratio : Percentage of team possessions that end in assists
    df['WAstR'] =  100 * df.WAst / (df.WFGA + 0.44*df.WFTA + df.WAst + df.WTO)
    df['LAstR'] = 100 * df.LAst / (df.LFGA + 0.44*df.LFTA + df.LAst + df.LTO)
    #Turnover Ratio: Number of turnovers of a team per 100 possessions used.
    #(TO * 100) / (FGA + (FTA * 0.44) + AST + TO)
    df['WTOR'] = 100 * df.WTO / (df.WFGA + 0.44*df.WFTA + df.WAst + df.WTO)
    df['LTOR'] = 100 * df.LTO / (df.LFGA + 0.44*df.LFTA + df.LAst + df.LTO)
                        
    #The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
    df['WTSP'] = 100 * df.WPts / (2 * (df.WFGA + 0.44 * df.WFTA))
    df['LTSP'] = 100 * df.LPts / (2 * (df.LFGA + 0.44 * df.LFTA))
    #eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable 
    df['WeFGP'] = (df.WFGM + 0.5 * df.WFGM3) / df.WFGA     
    df['LeFGP'] = (df.LFGM + 0.5 * df.LFGM3) / df.LFGA  
    #FTA Rate : How good a team is at drawing fouls.
    df['WFTAR'] = df.WFTA / df.WFGA
    df['LFTAR'] = df.LFTA / df.LFGA
                            
    #OREB% : Percentage of team offensive rebounds
    df['WORP'] = df.WOR / (df.WOR + df.LDR)
    df['LORP'] = df.LOR / (df.LOR + df.WDR)
    #DREB% : Percentage of team defensive rebounds
    df['WDRP'] = df.WDR / (df.WDR + df.LOR)
    df['LDRP'] = df.LDR / (df.LDR + df.WOR)                                     
    #REB% : Percentage of team total rebounds
    df['WRP'] = (df.WDR + df.WOR) / (df.WDR + df.WOR + df.LDR + df.LOR)
    df['LRP'] = (df.LDR + df.LOR) / (df.WDR + df.WOR + df.LDR + df.LOR)
    logger.info("Done advanced stats")
    return df

def calc_season_statistics(regular_data):
    """Calc season statistics using Kaggle data
    """
    logger.info("Calculating season statistics")

    exclude_cols = ['TeamID', 'Score', 'Loc']
    exclude_cols2 = ["Season", "DayNum", "NumOT", "location"]
    boxscore_cols = [c for c in regular_data.columns if c[3:] not in exclude_cols and c not in exclude_cols2]

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
    logger.info("Calculating 14 day win ratio")
    # Calc prior 2 weeks win %
    last14_days_T1 = regular_data.loc[regular_data['DayNum']>118].reset_index(drop=True)
    last14_days_T1['win'] = np.where(last14_days_T1['T1_PointDiff'] > 0, 1, 0)
    last14_days_T1 = last14_days_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

    last14_days_T2 = regular_data.loc[regular_data['DayNum']>118].reset_index(drop=True)
    last14_days_T2['win'] = np.where(last14_days_T2['T1_PointDiff'] < 0, 1, 0)
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

def build_test_data(data):
    proj_dir = Path().resolve().parents[0]
    data_dir = proj_dir / "data" 
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
    
    # feature eng:
    # season statistics:
    T1_season_stats, T2_season_stats = calc_season_statistics(regular_data)
    last14_days_T1, last14_days_T2 = win_ratio_14_days(regular_data)
    seeds_T1, seeds_T2 = calc_seed_diff(seeds)

    # combine:
    feature_set = data.merge(T1_season_stats, on=['Season', 'T1_TeamID'], how='left')
    feature_set = feature_set.merge(T2_season_stats, on=['Season', 'T2_TeamID'], how='left')
    feature_set = pd.merge(feature_set, last14_days_T1, on=['Season', 'T1_TeamID'], how='left')
    feature_set = pd.merge(feature_set, last14_days_T2, on=['Season', 'T2_TeamID'], how='left')
    feature_set = feature_set.merge(T1_kp, on=['Season', 'T1_TeamID'], how='left')
    feature_set = feature_set.merge(T2_kp, on=['Season', 'T2_TeamID'], how='left')
    feature_set = pd.merge(feature_set, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
    feature_set = pd.merge(feature_set, seeds_T2, on=['Season', 'T2_TeamID'], how='left')
    feature_set['SeedDiff'] = feature_set['T1_seed'] - feature_set['T2_seed']
    return feature_set


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

