import logging
import sys
import os

# data wrangling tools
import pandas as pd
import numpy as np

# datadotworld SDK
import datadotworld as ddw

import time

from Approx_Least_Square_Alg.commons import twoNorm, get_w, data_normalize_by_features
from numpy.linalg import inv as mat_inv


def setup_dataotworld_env():
    # Set environment variables
    os.environ['DW_AUTH_TOKEN'] = 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJweXRob246bXVuZXlvc2hpIiwiaXNzIjoiY2xp' \
                                  'ZW50OnB5dGhvbjphZ2VudDptdW5leW9zaGk6OmIzMzdlZWVhLWI2MTEtNGM3MC1hMWI3L' \
                                  'TUyNGQ4NmJmMjI5OCIsImlhdCI6MTY3MTQ3MzEwNiwicm9sZSI6WyJ1c2VyX2FwaV9hZG' \
                                  '1pbiIsInVzZXJfYXBpX3JlYWQiLCJ1c2VyX2FwaV93cml0ZSJdLCJnZW5lcmFsLXB1cn' \
                                  'Bvc2UiOnRydWUsInNhbWwiOnt9fQ.ZdtUxoOhjAcjTmzKBws2Vb98Am6RqYyRJ5Qf3Qka' \
                                  's5bdy_M7FkuMBqwgx0Ova82AbA8uJEUFsvv0UF7oYZngXA'

    # Get environment variables
    return os.getenv('DW_AUTH_TOKEN')


def retrieve_cancer_data():
    start = time.time()
    mortdf = ddw.query('nrippner/cancer-analysis-hackathon-challenge',
                       'SELECT * FROM `death .csv/death `').dataframe

    incddf = ddw.query('nrippner/cancer-analysis-hackathon-challenge',
                       'SELECT * FROM `incd.csv/incd`').dataframe
    end = time.time()
    logging.info(f"mortality data retrieving cost {end - start}")

    mortdf = mortdf[mortdf.fips.notnull()]
    incddf = incddf[incddf.fips.notnull()]

    mortdf['FIPS'] = mortdf.fips.apply(lambda x: str(int(x))) \
        .astype(np.object_) \
        .str.pad(5, 'left', '0')

    incddf['FIPS'] = incddf.fips.apply(lambda x: str(int(x))) \
        .astype(np.object_) \
        .str.pad(5, 'left', '0')

    incddf.drop(incddf.columns[[0, 3, 4, 7, 8, 9]].values, axis=1, inplace=True)
    mortdf.drop(mortdf.columns[[0, 2, 4, 5, 7, 8, 9, 10]], axis=1, inplace=True)

    incddf.rename(columns={incddf.columns[1]: 'Incidence_Rate',
                           incddf.columns[2]: 'Avg_Ann_Incidence'}, inplace=True)
    mortdf.rename(columns={mortdf.columns[1]: 'Mortality_Rate',
                           mortdf.columns[2]: 'Avg_Ann_Deaths'}, inplace=True)

    return incddf, mortdf


def retrieve_poverty_data():
    # Retrieve a list of table names (by state)
    pov = ddw.load_dataset('uscensusbureau/acs-2015-5-e-poverty')

    tables = []
    for i in pov.tables:
        if len(i) == 2:
            tables.append(i)

    # remove Puerto Rico
    tables.remove('pr')

    # Retrieve the Census poverty data from data.world
    start = time.time()

    # a string - the poverty columns we want from the Census ACS
    cols = '`State`, `StateFIPS`, `CountyFIPS`, `AreaName`, `B17001_002`, `B17001_003`,' \
           '`B17001_017`'

    # call the data for each state and concatenate
    for i, state in enumerate(tables):
        if i == 0:
            povdf = ddw.query('uscensusbureau/acs-2015-5-e-poverty',
                              '''SELECT %s FROM `AK`
                                 WHERE SummaryLevel=50''' % cols).dataframe
        else:
            df = ddw.query('uscensusbureau/acs-2015-5-e-poverty',
                           '''SELECT %s FROM `%s`
                              WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe

            povdf = pd.concat([povdf, df], ignore_index=True)

    end = time.time()
    logging.info(f"poverty data retrieving cost {end-start}")

    # Add leading zeros to the state and county FIPS codes
    povdf['StateFIPS'] = povdf.StateFIPS.astype(np.object_) \
        .apply(lambda x: str(x)) \
        .str.pad(2, 'left', '0')
    povdf['CountyFIPS'] = povdf.CountyFIPS.astype(np.object_) \
        .apply(lambda x: str(x)) \
        .str.pad(3, 'left', '0')

    povdf.rename(columns={'B17001_002': 'All_Poverty', 'B17001_003': 'M_Poverty', 'B17001_017': 'F_Poverty'},
                 inplace=True)

    return povdf


def retrieve_income_data():
    # Retrieve a list of table names (by state)
    pov = ddw.load_dataset('uscensusbureau/acs-2015-5-e-poverty')

    tables = []
    for i in pov.tables:
        if len(i) == 2:
            tables.append(i)

    # remove Puerto Rico
    tables.remove('pr')

    cols = '`StateFIPS`, `CountyFIPS`,' \
           '`B19013_001`, `B19013A_001`, `B19013B_001`, `B19013C_001`, `B19013D_001`,' \
           '`B19013I_001`'

    start = time.time()
    for i, state in enumerate(tables):
        if i == 0:
            incomedf = ddw.query('uscensusbureau/acs-2015-5-e-income',
                                 '''SELECT %s FROM `AK`
                                    WHERE SummaryLevel=50''' % cols).dataframe
        else:
            df = ddw.query('uscensusbureau/acs-2015-5-e-income',
                           '''SELECT %s FROM `%s`
                              WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe
            incomedf = pd.concat([incomedf, df], ignore_index=True)

    end = time.time()
    logging.info(f"income data retrieving cost {end - start}")

    incomedf['StateFIPS'] = incomedf.StateFIPS.astype(np.object_) \
        .apply(lambda x: str(x)) \
        .str.pad(2, 'left', '0')
    incomedf['CountyFIPS'] = incomedf.CountyFIPS.astype(np.object_) \
        .apply(lambda x: str(x)) \
        .str.pad(3, 'left', '0')

    incomedf.rename(columns={'B19013_001': 'Med_Income', 'B19013A_001': 'Med_Income_White',
                             'B19013B_001': 'Med_Income_Black', 'B19013C_001': 'Med_Income_Nat_Am',
                             'B19013D_001': 'Med_Income_Asian', 'B19013I_001': 'Hispanic'}, inplace=True)

    return incomedf


def retrieve_health_insurance_data():
    # Retrieve a list of table names (by state)
    pov = ddw.load_dataset('uscensusbureau/acs-2015-5-e-poverty')

    tables = []
    for i in pov.tables:
        if len(i) == 2:
            tables.append(i)

    # remove Puerto Rico
    tables.remove('pr')


    cols = '`StateFIPS`, `CountyFIPS`,' \
           '`B27001_004`, `B27001_005`, `B27001_007`, `B27001_008`,' \
           '`B27001_010`, `B27001_011`, `B27001_013`, `B27001_014`,' \
           '`B27001_016`, `B27001_017`, `B27001_019`, `B27001_020`,' \
           '`B27001_022`, `B27001_023`, `B27001_025`, `B27001_026`,' \
           '`B27001_028`, `B27001_029`, `B27001_032`, `B27001_033`,' \
           '`B27001_035`, `B27001_036`, `B27001_038`, `B27001_039`,' \
           '`B27001_041`, `B27001_042`, `B27001_044`, `B27001_045`,' \
           '`B27001_047`, `B27001_048`, `B27001_050`, `B27001_051`,' \
           '`B27001_053`, `B27001_054`, `B27001_056`, `B27001_057`'
    # male <= 029

    start = time.time()
    for i, state in enumerate(tables):
        if i == 0:
            hinsdf = ddw.query('uscensusbureau/acs-2015-5-e-healthinsurance',
                               '''SELECT %s FROM `AK`
                                  WHERE SummaryLevel=50''' % cols).dataframe

        else:
            df = ddw.query('uscensusbureau/acs-2015-5-e-healthinsurance',
                           '''SELECT %s FROM `%s`
                              WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe
            hinsdf = pd.concat([hinsdf, df], ignore_index=True)

    end = time.time()
    logging.info(f"health_insurance data retrieving cost {end - start}")

    hinsdf['StateFIPS'] = hinsdf.StateFIPS.astype(np.object_) \
        .apply(lambda x: str(x)) \
        .str.pad(2, 'left', '0')
    hinsdf['CountyFIPS'] = hinsdf.CountyFIPS.astype(np.object_) \
        .apply(lambda x: str(x)) \
        .str.pad(3, 'left', '0')

    # columns representing males' health insurance statistics
    males = ['`B27001_004`', '`B27001_005`', '`B27001_007`', '`B27001_008`',
             '`B27001_010`', '`B27001_011`', '`B27001_013`', '`B27001_014`',
             '`B27001_016`', '`B27001_017`', '`B27001_019`', '`B27001_020`',
             '`B27001_022`', '`B27001_023`', '`B27001_025`', '`B27001_026`',
             '`B27001_028`', '`B27001_029`']

    # females' health insurance statistics
    females = ['`B27001_032`', '`B27001_033`', '`B27001_035`', '`B27001_036`',
               '`B27001_038`', '`B27001_039`', '`B27001_041`', '`B27001_042`',
               '`B27001_044`', '`B27001_045`', '`B27001_047`', '`B27001_048`',
               '`B27001_050`', '`B27001_051`', '`B27001_053`', '`B27001_054`',
               '`B27001_056`', '`B27001_057`']

    # separate the "with" and "without" health insurance columns
    males_with = []
    males_without = []
    females_with = []
    females_without = []

    # strip the backticks
    for i, j in enumerate(males):
        if i % 2 == 0:
            males_with.append(j.replace('`', ''))
        else:
            males_without.append(j.replace('`', ''))

    for i, j in enumerate(females):
        if i % 2 == 0:
            females_with.append(j.replace('`', ''))
        else:
            females_without.append(j.replace('`', ''))

    # Create features that sum all the individual age group
    newcols = ['M_With', 'M_Without', 'F_With', 'F_Without']

    for col in newcols:
        hinsdf[col] = 0

    for i in males_with:
        hinsdf['M_With'] += hinsdf[i]
    for i in males_without:
        hinsdf['M_Without'] += hinsdf[i]
    for i in females_with:
        hinsdf['F_With'] += hinsdf[i]
    for i in females_without:
        hinsdf['F_Without'] += hinsdf[i]

    hinsdf['All_With'] = hinsdf.M_With + hinsdf.F_With
    hinsdf['All_Without'] = hinsdf.M_Without + hinsdf.F_Without

    hinsdf.drop(df.columns[df.columns.str.contains('B27001')].values, axis=1, inplace=True)

    return hinsdf


def retrieve_population_data():
    start = time.time()
    populationdf = ddw.query('nrippner/us-population-estimates-2015',
                             '''SELECT `POPESTIMATE2015`, `STATE`, `COUNTY`
                                FROM `CO-EST2015-alldata`''').dataframe

    state = populationdf.STATE.apply(lambda x: str(x)) \
        .str.pad(2, 'left', '0')
    county = populationdf.COUNTY.apply(lambda x: str(x)) \
        .str.pad(3, 'left', '0')

    populationdf['FIPS'] = state + county
    end = time.time()
    logging.info(f"population data retrieving cost {end - start}")

    return populationdf


def merge_dataframe(povdf, incomedf, hinsdf, incddf, mortdf, populationdf):
    # create FIPS features
    for df in [povdf, incomedf, hinsdf]:
        df['FIPS'] = df.StateFIPS + df.CountyFIPS
        df.drop(['StateFIPS', 'CountyFIPS'], axis=1, inplace=True)

    # then merge
    dfs = [povdf, incomedf, hinsdf, incddf, mortdf]

    fulldf = None
    for i, j in enumerate(dfs):
        if i == 0:
            fulldf = j.copy()
        else:
            fulldf = fulldf.merge(j, how='inner', on='FIPS')

    fulldf.drop(['Med_Income_White', 'Med_Income_Black', 'Med_Income_Nat_Am',
                 'Med_Income_Asian', 'Hispanic', 'fips_x', 'fips_y'], axis=1, inplace=True)

    #   further merge
    fulldf = fulldf.merge(populationdf[['FIPS', 'POPESTIMATE2015']], on='FIPS', how='inner')

    #   filter out the records with missing mortality rate values.
    fulldf = fulldf[fulldf.Mortality_Rate != '*']

    #   clean up Med_Income
    fulldf['Med_Income'] = pd.to_numeric(fulldf.Med_Income)

    #   rename 'Recent Trend' to remove the space
    fulldf.rename(columns={'recent_trend': 'RecentTrend'}, inplace=True)

    #   change all the missing values to the mode ('stable')
    fulldf.replace({'RecentTrend': {'*': 'stable'}}, inplace=True)

    #   function to do boolean check and return 1 or 0
    def f(x, term):
        if x == term:
            return 1
        else:
            return 0
    # create new features using the apply method with the 'f' function we defined above
    fulldf['Rising'] = fulldf.RecentTrend.apply(lambda x: f(x, term='rising'))
    fulldf['Falling'] = fulldf.RecentTrend.apply(lambda x: f(x, term='falling'))

    return fulldf


def produce_data_for_LS(fulldf):
    y = pd.to_numeric(fulldf.Mortality_Rate).values
    X = fulldf.loc[:, ['All_Poverty', 'M_Poverty', 'F_Poverty', 'Med_Income',
                       'M_With', 'M_Without', 'F_With', 'F_Without', 'All_With',
                       'All_Without', 'Incidence_Rate', 'Falling', 'Rising',
                       'POPESTIMATE2015']]

    #   The "errors='coerce'" argument to the pd.to_numeric method,
    #   tells pandas to replace any value that can't be converted to a float with 'nan'.
    X['Incidence_Rate'] = pd.to_numeric(X.Incidence_Rate, errors='coerce')
    X['Incidence_Rate'] = X.Incidence_Rate.fillna(X.Incidence_Rate.median())

    #   Now let's convert the applicable variables to per capita.
    #   We'll retain the original features and name the per capita versions with the "_PC" suffix.
    for col in ['All_Poverty', 'M_Poverty', 'F_Poverty', 'M_With',
                'M_Without', 'F_With', 'F_Without', 'All_With', 'All_Without']:
        X[col + "_PC"] = X[col] / X.POPESTIMATE2015 * 10 ** 5

    #   We see that All_Poverty_PC, F_Poverty_PC and M_Poverty_PC look pretty much perfectly correlated.
    #   We don't want redundant features. So let's drop M_Poverty_PC and F_Poverty_PC and keep only
    #   All_Poverty_PC to represent poverty information in our model.
    X.drop(['M_Poverty_PC', 'F_Poverty_PC'], axis=1, inplace=True)

    #   repeating this process with our health insurance variables
    X.drop(['M_With_PC', 'F_With_PC'], axis=1, inplace=True)
    X.drop(['M_Without_PC', 'F_Without_PC'], axis=1, inplace=True)

    cols = ['All_Poverty_PC', 'Med_Income', 'All_With_PC', 'All_Without_PC',
            'Incidence_Rate', 'POPESTIMATE2015', 'Falling', 'Rising', 'All_Poverty',
            'All_With', 'All_Without']
    # add constant (coloumn vector of all 1s)
    X = X[cols]
    X['Constant'] = 1
    X.reset_index(drop=True, inplace=True)

    return X, y


def save_cancer_dataset_matrices_for_LS(X, y, file_X_name="./Dataset/cancer-LR-X.txt",
                                        file_y_name="./Dataset/cancer-LR-y.txt"):
    np.savetxt(file_y_name, y, delimiter=',')
    np.savetxt(file_X_name, X.to_numpy(), delimiter=',')


def load_cancer_dataset_matrices_for_LS(file_X_name="./Dataset/cancer-LR-X.txt",
                                        file_y_name="./Dataset/cancer-LR-y.txt"):
    B = np.loadtxt(file_X_name, delimiter=',')  # B is an array
    b = np.loadtxt(file_y_name, delimiter=',')  # b is an array
    return B, b


def main_generate_cancer_matrices_for_LS(file_X_name="./Dataset/cancer-LR-X.txt",
                                        file_y_name="./Dataset/cancer-LR-y.txt"):
    setup_dataotworld_env()
    incddf, mortdf = retrieve_cancer_data()
    povdf = retrieve_poverty_data()
    incomedf = retrieve_income_data()
    hinsdf = retrieve_health_insurance_data()
    populationdf = retrieve_population_data()
    fulldf = merge_dataframe(povdf, incomedf, hinsdf, incddf, mortdf, populationdf)
    X, y = produce_data_for_LS(fulldf)
    save_cancer_dataset_matrices_for_LS(X, y, file_X_name, file_y_name)

    logging.info("generation completes")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler("./dataset_generating.log"),
            logging.StreamHandler()
        ]
    )

    file_X_name = "./cancer-LR-X.txt"
    file_y_name = "./cancer-LR-y.txt"
    try:
        B, b = load_cancer_dataset_matrices_for_LS(file_X_name, file_y_name)
    except:
        main_generate_cancer_matrices_for_LS(file_X_name, file_y_name)
        B, b = load_cancer_dataset_matrices_for_LS(file_X_name, file_y_name)

    B, b = data_normalize_by_features(B, b)
    ret_1 = twoNorm(get_w(B, b))

    print(ret_1)

    M = mat_inv(B.T @ B)
    P = B @ M @ B.T
    ret_2 = twoNorm((np.identity(B.shape[0]) - P) @ b)

    print(ret_2)




