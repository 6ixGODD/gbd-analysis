from __future__ import annotations

import pathlib
import typing

import joblib
import pandas as pd
from tabulate import tabulate

from gbd.common import (
    const as _const,
    utils as _utils,
)

SPEC_COLUMNS = ['year', 'age_name', 'sex_name', 'location_name']

LOGGER = _utils.Logger()


def calc_percent_change(old: float, new: float, /) -> float:
    try:
        return ((new - old) / old) * 100
    except ZeroDivisionError:
        LOGGER.warning(f"Division by zero: {old=}, {new=}")
        return float('inf')


def calc_asr(
    raw: pd.DataFrame,
    *,
    std_pop: pd.DataFrame,
    per: float = 1e3
) -> typing.Tuple[float, float, float]:
    """
    Calculate Age-Standardized Rate (ASR).

    Args:
        raw (pd.DataFrame): GBD data.
        std_pop (pd.DataFrame): Standard population.
                                Should have columns 'age_name' and 'standard_population'.
        per (int): Population to standardize against. Defaults to 1e3.

    Returns:
        Tuple[float, float, float]: ASR, lower bound, and upper bound.
    """
    raw = raw.copy().merge(std_pop, on='age_name', how='left')
    total_standard_pop = raw['standard_population'].sum()

    # Val
    raw['rate'] = raw['val_num'] / raw['population'] * per
    raw['weighted_rate'] = raw['rate'] * raw['standard_population']
    asir = raw['weighted_rate'].sum() / total_standard_pop

    # Lower
    raw['rate_lower'] = raw['lower_num'] / raw['population'] * per
    raw['weighted_rate_lower'] = raw['rate_lower'] * raw['standard_population']
    asir_lower = raw['weighted_rate_lower'].sum() / total_standard_pop

    # Upper
    raw['rate_upper'] = raw['upper_num'] / raw['population'] * per
    raw['weighted_rate_upper'] = raw['rate_upper'] * raw['standard_population']
    asir_upper = raw['weighted_rate_upper'].sum() / total_standard_pop

    return asir, asir_lower, asir_upper


def run_stat(
    incidence_df: pd.DataFrame,
    *,
    years: typing.Tuple[int, int],
    std_pop: pd.DataFrame,
    cause: str
) -> pd.DataFrame:
    """ASIR and Number Statistics."""
    y1, y2 = years
    LOGGER.info(f"Calculating ASIR for {cause} from {y1} to {y2}")
    stat_results = pd.DataFrame(
        columns=['Characteristic', f'Number * 1e6 in {y1} (95% UI)', f'Number * 1e6 in {y2} (95% UI)',
                 'Percentage Change of Number (%, 95% UI)', f'ASIR per 1e3 population in {y1} (95% UI)',
                 f'ASIR per 1e3 population in {y2} (95% UI)', 'Percentage Change of ASIR (%, 95% UI)']
    )
    for spec in ['global', 'sex_name', 'location_name']:
        # Global
        if spec == 'global':
            incidence_df = incidence_df[(incidence_df['sex_name'] == 'Both') & (incidence_df['location_name'] == 'Global')]
            # Calculate total cases
            incidence_cases_year = incidence_df.copy().groupby('year').agg(_const.AGG_TEMPLATE).reset_index()
            incidence_cases_year['val_num'] /= 1e6  # num * 1e6
            incidence_cases_year['upper_num'] /= 1e6
            incidence_cases_year['lower_num'] /= 1e6

            LOGGER.info("=== Global ===")
            LOGGER.info(
                f"Total cases in {y1}: "
                f"{incidence_cases_year[incidence_cases_year['year'] == y1]['val_num'].iloc[0]:.2f}"
            )
            LOGGER.info(
                f"95% UI: ({incidence_cases_year[incidence_cases_year['year'] == y1]['lower_num'].iloc[0]:.2f}, "
                f"{incidence_cases_year[incidence_cases_year['year'] == y1]['upper_num'].iloc[0]:.2f})"
            )
            LOGGER.info(
                f"Total cases in {y2}: "
                f"{incidence_cases_year[incidence_cases_year['year'] == y2]['val_num'].iloc[0]:.2f}"
            )
            LOGGER.info(
                f"95% UI: ({incidence_cases_year[incidence_cases_year['year'] == y2]['lower_num'].iloc[0]:.2f}, "
                f"{incidence_cases_year[incidence_cases_year['year'] == y2]['upper_num'].iloc[0]:.2f})"
            )

            # Calculate percent change
            incidence_cases_y1 = incidence_cases_year[incidence_cases_year['year'] == y1].iloc[0]
            incidence_cases_y2 = incidence_cases_year[incidence_cases_year['year'] == y2].iloc[0]
            percent_change_val = calc_percent_change(incidence_cases_y1['val_num'], incidence_cases_y2['val_num'])
            percent_change_upper = calc_percent_change(incidence_cases_y1['upper_num'], incidence_cases_y2['upper_num'])
            percent_change_lower = calc_percent_change(incidence_cases_y1['lower_num'], incidence_cases_y2['lower_num'])

            LOGGER.info(f"Percentage change in cases from {y1} to {y2}: {percent_change_val:.2f}%")
            LOGGER.info(f"95% UI: ({percent_change_lower:.2f}%, {percent_change_upper:.2f}%)")

            # Calculate ASIR (per 1e3 population)
            incidence_merged_ = incidence_df.copy()
            cases_age_y1 = incidence_merged_[incidence_merged_['year'] == y1].groupby('age_name').agg(
                _const.AGG_TEMPLATE
            ).reset_index()
            cases_age_y2 = incidence_merged_[incidence_merged_['year'] == y2].groupby('age_name').agg(
                _const.AGG_TEMPLATE
            ).reset_index()
            asir1, asir_lower1, asir_upper1 = calc_asr(cases_age_y1, std_pop=std_pop)
            asir2, asir_lower2, asir_upper2 = calc_asr(cases_age_y2, std_pop=std_pop)

            LOGGER.info(f"ASIR per 1e3 population in {y1}: {asir1:.2f}")
            LOGGER.info(f"95% UI: ({asir_lower1:.2f}, {asir_upper1:.2f})")
            LOGGER.info(f"ASIR per 1e3 population in {y2}: {asir2:.2f}")
            LOGGER.info(f"95% UI: ({asir_lower2:.2f}, {asir_upper2:.2f})")

            # Calculate percent change in ASIR
            percent_change_asir = calc_percent_change(asir1, asir2)
            percent_change_asir_lower = calc_percent_change(asir_lower1, asir_lower2)
            percent_change_asir_upper = calc_percent_change(asir_upper1, asir_upper2)

            LOGGER.info(f"Percentage change in ASIR from {y1} to {y2}: {percent_change_asir:.2f}%")
            LOGGER.info(f"95% UI: ({percent_change_asir_lower:.2f}%, {percent_change_asir_upper:.2f}%)")

            stat_results = pd.concat(
                [stat_results, pd.DataFrame(
                    {
                        'Characteristic':                            'Global',
                        f'Number * 1e6 in {y1} (95% UI)':            f"{incidence_cases_y1['val_num']:.2f} ("
                                                                     f"{incidence_cases_y1['lower_num']:.2f}, "
                                                                     f"{incidence_cases_y1['upper_num']:.2f})",
                        f'Number * 1e6 in {y2} (95% UI)':            f"{incidence_cases_y2['val_num']:.2f} ("
                                                                     f"{incidence_cases_y2['lower_num']:.2f}, "
                                                                     f"{incidence_cases_y2['upper_num']:.2f})",
                        'Percentage Change of Number (%, 95% UI)':   f"{percent_change_val:.2f} ("
                                                                     f"{percent_change_lower:.2f}, "
                                                                     f"{percent_change_upper:.2f})",
                        f'ASIR per 1e3 population in {y1} (95% UI)': f"{asir1:.2f} ({asir_lower1:.2f}, "
                                                                     f"{asir_upper1:.2f})",
                        f'ASIR per 1e3 population in {y2} (95% UI)': f"{asir2:.2f} ({asir_lower2:.2f}, "
                                                                     f"{asir_upper2:.2f})",
                        'Percentage Change of ASIR (%, 95% UI)':     f"{percent_change_asir:.2f} ("
                                                                     f"{percent_change_asir_lower:.2f}, "
                                                                     f"{percent_change_asir_upper:.2f})"
                    },
                    index=[0]
                )],
                ignore_index=True
            )
        # Sex-specific and Location-specific
        else:
            incidence_cases_year = incidence_df.copy().groupby(['year', spec]).agg(_const.AGG_TEMPLATE).reset_index()
            incidence_cases_year['val_num'] /= 1e6
            incidence_cases_year['upper_num'] /= 1e6
            incidence_cases_year['lower_num'] /= 1e6

            # Each item of the characteristic
            for item in incidence_cases_year[spec].unique():
                cases_spec = incidence_cases_year[incidence_cases_year[spec] == item]
                incidence_cases_y1 = cases_spec[cases_spec['year'] == y1].iloc[0]
                incidence_cases_y2 = cases_spec[cases_spec['year'] == y2].iloc[0]
                percent_change_val = calc_percent_change(incidence_cases_y1['val_num'], incidence_cases_y2['val_num'])
                percent_change_upper = calc_percent_change(
                    incidence_cases_y1['upper_num'],
                    incidence_cases_y2['upper_num']
                )
                percent_change_lower = calc_percent_change(
                    incidence_cases_y1['lower_num'],
                    incidence_cases_y2['lower_num']
                )

                LOGGER.info(f"=== {spec} - {item} ===")
                LOGGER.info(f"Total cases in {item} of {spec} in {y1}: {incidence_cases_y1['val_num']:.2f}")
                LOGGER.info(
                    f"95% UI: ({incidence_cases_y1['lower_num']:.2f}, {incidence_cases_y1['upper_num']:.2f})"
                )
                LOGGER.info(f"Total cases in {item} of {spec} in {y2}: {incidence_cases_y2['val_num']:.2f}")
                LOGGER.info(
                    f"95% UI: ({incidence_cases_y2['lower_num']:.2f}, {incidence_cases_y2['upper_num']:.2f})"
                )
                LOGGER.info(f"Percentage change in cases from {y1} to {y2} of {spec}: {percent_change_val:.2f}%")
                LOGGER.info(f"95% UI: ({percent_change_lower:.2f}%, {percent_change_upper:.2f}%)")

                # Calculate ASIR (per 1e3 population)
                incidence_merged_ = incidence_df.copy()[incidence_df[spec] == item]
                cases_age_y1 = incidence_merged_[incidence_merged_['year'] == y1].groupby('age_name').agg(
                    _const.AGG_TEMPLATE
                ).reset_index()
                cases_age_y2 = incidence_merged_[incidence_merged_['year'] == y2].groupby('age_name').agg(
                    _const.AGG_TEMPLATE
                ).reset_index()
                asir1, asir_lower1, asir_upper1 = calc_asr(cases_age_y1, std_pop=std_pop)
                asir2, asir_lower2, asir_upper2 = calc_asr(cases_age_y2, std_pop=std_pop)

                LOGGER.info(f"ASIR per 1e3 population in {y1}: {asir1:.2f}")
                LOGGER.info(f"95% UI: ({asir_lower1:.2f}, {asir_upper1:.2f})")
                LOGGER.info(f"ASIR per 1e3 population in {y2}: {asir2:.2f}")
                LOGGER.info(f"95% UI: ({asir_lower2:.2f}, {asir_upper2:.2f})")

                # Calculate percent change in ASIR
                percent_change_asir = calc_percent_change(asir1, asir2)
                percent_change_asir_lower = calc_percent_change(asir_lower1, asir_lower2)
                percent_change_asir_upper = calc_percent_change(asir_upper1, asir_upper2)

                LOGGER.info(f"Percentage change in ASIR from {y1} to {y2}: {percent_change_asir:.2f}%")
                LOGGER.info(f"95% UI: ({percent_change_asir_lower:.2f}%, {percent_change_asir_upper:.2f}%)")

                stat_results = pd.concat(
                    [stat_results, pd.DataFrame(
                        {
                            'Characteristic':                            f"{spec} - {item}",
                            f'Number * 1e6 in {y1} (95% UI)':            f"{incidence_cases_y1['val_num']:.2f} ("
                                                                         f"{incidence_cases_y1['lower_num']:.2f}, "
                                                                         f"{incidence_cases_y1['upper_num']:.2f})",
                            f'Number * 1e6 in {y2} (95% UI)':            f"{incidence_cases_y2['val_num']:.2f} ("
                                                                         f"{incidence_cases_y2['lower_num']:.2f}, "
                                                                         f"{incidence_cases_y2['upper_num']:.2f})",
                            'Percentage Change of Number (%, 95% UI)':   f"{percent_change_val:.2f} ("
                                                                         f"{percent_change_lower:.2f}, "
                                                                         f"{percent_change_upper:.2f})",
                            f'ASIR per 1e3 population in {y1} (95% UI)': f"{asir1:.2f} ({asir_lower1:.2f}, "
                                                                         f"{asir_upper1:.2f})",
                            f'ASIR per 1e3 population in {y2} (95% UI)': f"{asir2:.2f} ({asir_lower2:.2f}, "
                                                                         f"{asir_upper2:.2f})",
                            'Percentage Change of ASIR (%, 95% UI)':     f"{percent_change_asir:.2f} ("
                                                                         f"{percent_change_asir_lower:.2f}, "
                                                                         f"{percent_change_asir_upper:.2f})"
                        },
                        index=[0]
                    )],
                    ignore_index=True
                )
    LOGGER.info("=== Results ===")
    print(tabulate(stat_results, headers='keys', tablefmt='pretty'))
    return stat_results


def run_year_trend(
    incidence_df: pd.DataFrame,
    daly_df: pd.DataFrame,
    prevalence_df: pd.DataFrame,
    *,
    years: typing.Tuple[int, int],
    std_pop: pd.DataFrame,
    cause: str
) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ASIR, ASDR, and ASPR Year Trend. for Joinpoint Analysis."""
    asir_results = pd.DataFrame(columns=['Year', 'ASIR', 'ASIR Lower', 'ASIR Upper', 'Standard Error'])
    asdr_results = pd.DataFrame(columns=['Year', 'ASDR', 'ASDR Lower', 'ASDR Upper', 'Standard Error'])
    aspr_results = pd.DataFrame(columns=['Year', 'ASPR', 'ASPR Lower', 'ASPR Upper', 'Standard Error'])

    for year in range(years[0], years[1] + 1):
        LOGGER.info(f"Calculating ASIR, ASDR, and ASPR for {cause} in {year}")
        incidence_year = incidence_df[incidence_df['year'] == year]
        daly_year = daly_df[daly_df['year'] == year]
        prevalence_year = prevalence_df[prevalence_df['year'] == year]

        # ASIR
        cases_age = incidence_year.groupby('age_name').agg(_const.AGG_TEMPLATE).reset_index()
        asir, asir_lower, asir_upper = calc_asr(cases_age, std_pop=std_pop, per=1e5)
        asir_results = pd.concat(
            [asir_results, pd.DataFrame(
                {
                    'Year':           year,
                    'ASIR':           asir,
                    'ASIR Lower':     asir_lower,
                    'ASIR Upper':     asir_upper,
                    'Standard Error': (asir_upper - asir_lower) / (1.96 * 2)  # 95% UI
                },
                index=[0]
            )],
            ignore_index=True
        )

        # ASDR
        daly_age = daly_year.groupby('age_name').agg(_const.AGG_TEMPLATE).reset_index()
        asdr, asdr_lower, asdr_upper = calc_asr(daly_age, std_pop=std_pop, per=1e5)
        asdr_results = pd.concat(
            [asdr_results, pd.DataFrame(
                {
                    'Year':           year,
                    'ASDR':           asdr,
                    'ASDR Lower':     asdr_lower,
                    'ASDR Upper':     asdr_upper,
                    'Standard Error': (asdr_upper - asdr_lower) / (1.96 * 2)  # 95% UI
                },
                index=[0]
            )],
            ignore_index=True
        )

        # ASPR
        prevalence_age = prevalence_year.groupby('age_name').agg(_const.AGG_TEMPLATE).reset_index()
        aspr, aspr_lower, aspr_upper = calc_asr(prevalence_age, std_pop=std_pop, per=1e5)
        aspr_results = pd.concat(
            [aspr_results, pd.DataFrame(
                {
                    'Year':           year,
                    'ASPR':           aspr,
                    'ASPR Lower':     aspr_lower,
                    'ASPR Upper':     aspr_upper,
                    'Standard Error': (aspr_upper - aspr_lower) / (1.96 * 2)  # 95% UI
                },
                index=[0]
            )],
            ignore_index=True
        )

    LOGGER.info("=== ASIR Results ===")
    print(tabulate(asir_results, headers='keys', tablefmt='pretty'))

    LOGGER.info("=== ASDR Results ===")
    print(tabulate(asdr_results, headers='keys', tablefmt='pretty'))

    LOGGER.info("=== ASPR Results ===")
    print(tabulate(aspr_results, headers='keys', tablefmt='pretty'))

    return asir_results, asdr_results, aspr_results


def run_location_asdr(
    daly_df: pd.DataFrame,
    *,
    years: typing.Tuple[int, int],
    std_pop: pd.DataFrame,
    per: float = 1e5  # per 1e5 population
) -> pd.DataFrame:
    """ASDR for Each Location."""
    y1, y2 = years
    LOGGER.info(f"Calculating ASDR for each location from {y1} to {y2}")
    asdr_results = pd.DataFrame(columns=['Location', 'Year', 'ASDR', 'ASDR Lower', 'ASDR Upper', 'Standard Error'])

    for year in range(y1, y2 + 1):
        LOGGER.info(f"=== {year} ===")
        daly_year = daly_df[daly_df['year'] == year]
        for location in daly_year['location_name'].unique():
            daly_loc = daly_year[daly_year['location_name'] == location]
            daly_age = daly_loc.groupby('age_name').agg(_const.AGG_TEMPLATE).reset_index()
            asdr, asdr_lower, asdr_upper = calc_asr(daly_age, std_pop=std_pop, per=per)
            asdr_results = pd.concat(
                [asdr_results, pd.DataFrame(
                    {
                        'Location':       location,
                        'Year':           year,
                        'ASDR':           asdr,
                        'ASDR Lower':     asdr_lower,
                        'ASDR Upper':     asdr_upper,
                        'Standard Error': (asdr_upper - asdr_lower) / (1.96 * 2)  # 95% UI
                    },
                    index=[0]
                )],
                ignore_index=True
            )

            LOGGER.info(f"{location}: {asdr:.2f} ({asdr_lower:.2f}, {asdr_upper:.2f})")

    LOGGER.info("=== Results ===")
    print(tabulate(asdr_results, headers='keys', tablefmt='pretty'))
    return asdr_results


def run(
    raw: pd.DataFrame,
    *,
    std_pop: pd.DataFrame,
    years: typing.Tuple[int, int] = (1990, 2019),
    cause: str = 'Anxiety disorders',
    output_dir: pathlib.Path = pathlib.Path("output"),
    name: str = 'exp'
):
    # Initialize -------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _utils.unique_path(output_dir, name)

    # Incidence Data ---------------------------------------------------------
    # number
    incidence_number_df = raw[(raw['metric_name'] == 'Number') &  # number of cases
                              (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                              (raw['year'].isin(years)) &  # years
                              (raw['cause_name'] == cause) &  # cause
                              (raw['measure_name'] == 'Incidence')]  # incidence
    incidence_number_df = incidence_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    # rate
    incidence_rate_df = raw[(raw['metric_name'] == 'Rate') &  # rate of cases per 1e6
                            (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                            (raw['year'].isin(years)) &  # years
                            (raw['cause_name'] == cause) &  # cause
                            (raw['measure_name'] == 'Incidence')]  # incidence
    incidence_rate_df = incidence_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    # merge
    incidence_merged = incidence_number_df.merge(
        incidence_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    # per 1e5 population
    incidence_merged['population'] = incidence_merged['val_num'] / (incidence_merged['val_rate'] / 1e5)

    # Number and ASIR Statistics ---------------------------------------------
    stat_results = run_stat(
        incidence_merged,
        years=years,
        std_pop=std_pop,
        cause=cause
    )
    # save
    (output_dir / f'Number and ASIR of {cause}').mkdir(parents=True, exist_ok=True)
    stat_results.to_csv(output_dir / f'Number and ASIR of {cause}' / 'result.csv', index=False)
    LOGGER.info(f"Results saved to {output_dir / f'Number and ASIR of {cause}' / 'result.csv'}")

    # Year Trend -------------------------------------------------------------
    # Incidence Data
    incidence_number_df = raw[(raw['metric_name'] == 'Number') &  # number of cases
                              (raw['location_name'] == 'Global') &  # global
                              (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                              (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                              (raw['cause_name'] == cause) &  # cause
                              (raw['sex_name'] == 'Both') &  # only 'Both'
                              (raw['measure_name'] == 'Incidence')]  # incidence
    incidence_number_df = incidence_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    incidence_rate_df = raw[(raw['metric_name'] == 'Rate') &  # rate of cases per 1e6
                            (raw['location_name'] == 'Global') &  # global
                            (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                            (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                            (raw['cause_name'] == cause) &  # cause
                            (raw['sex_name'] == 'Both') &  # only 'Both'
                            (raw['measure_name'] == 'Incidence')]  # incidence
    incidence_rate_df = incidence_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    incidence_merged = incidence_number_df.merge(
        incidence_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    incidence_merged['population'] = incidence_merged['val_num'] / (incidence_merged['val_rate'] / 1e5)

    # DALYs Data
    dalys_number_df = raw[(raw['metric_name'] == 'Number') &  # number of DALYs
                          (raw['location_name'] == 'Global') &  # global
                          (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                          (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                          (raw['cause_name'] == cause) &  # cause
                          (raw['sex_name'] == 'Both') &  # only 'Both'
                          (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')]  # DALYs
    dalys_number_df = dalys_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    dalys_rate_df = raw[(raw['metric_name'] == 'Rate') &  # rate of DALYs per 1e6
                        (raw['location_name'] == 'Global') &  # global
                        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                        (raw['cause_name'] == cause) &  # cause
                        (raw['sex_name'] == 'Both') &  # only 'Both'
                        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')]  # DALYs
    dalys_rate_df = dalys_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    dalys_merged = dalys_number_df.merge(
        dalys_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    dalys_merged['population'] = dalys_merged['val_num'] / (dalys_merged['val_rate'] / 1e5)

    # Prevalence Data
    prevalence_number_df = raw[(raw['metric_name'] == 'Number') &  # number of cases
                               (raw['location_name'] == 'Global') &  # global
                               (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                               (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                               (raw['cause_name'] == cause) &  # cause
                               (raw['sex_name'] == 'Both') &  # only 'Both'
                               (raw['measure_name'] == 'Prevalence')]  # prevalence
    prevalence_number_df = prevalence_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    prevalence_rate_df = raw[(raw['metric_name'] == 'Rate') &  # rate of cases per 1e6
                             (raw['location_name'] == 'Global') &  # global
                             (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                             (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                             (raw['cause_name'] == cause) &  # cause
                             (raw['sex_name'] == 'Both') &  # only 'Both'
                             (raw['measure_name'] == 'Prevalence')]  # prevalence
    prevalence_rate_df = prevalence_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    prevalence_merged = prevalence_number_df.merge(
        prevalence_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    prevalence_merged['population'] = prevalence_merged['val_num'] / (prevalence_merged['val_rate'] / 1e5)

    # Year Trend
    asir_results, asdr_results, aspr_results = run_year_trend(
        incidence_merged,
        dalys_merged,
        prevalence_merged,
        years=years,
        std_pop=std_pop,
        cause=cause
    )
    # save
    (output_dir / 'Year Trend' / 'ASIR').mkdir(parents=True, exist_ok=True)
    (output_dir / 'Year Trend' / 'ASDR').mkdir(parents=True, exist_ok=True)
    (output_dir / 'Year Trend' / 'ASPR').mkdir(parents=True, exist_ok=True)
    asir_results.to_csv(output_dir / 'Year Trend' / 'ASIR' / 'trend.asir.csv', index=False)
    asdr_results.to_csv(output_dir / 'Year Trend' / 'ASDR' / 'trend.asdr.csv', index=False)
    aspr_results.to_csv(output_dir / 'Year Trend' / 'ASPR' / 'trend.aspr.csv', index=False)
    LOGGER.info(f"Results saved to {output_dir / 'Year Trend'}")

    # Location ASDR ----------------------------------------------------------
    # DALYs Data
    dalys_number_df = raw[(raw['metric_name'] == 'Number') &  # number of DALYs
                          (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                          (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                          (raw['cause_name'] == cause) &  # cause
                          (raw['sex_name'] == 'Both') &  # only 'Both'
                          (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')]  # DALYs
    dalys_number_df = dalys_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    dalys_rate_df = raw[(raw['metric_name'] == 'Rate') &  # rate of DALYs per 1e6
                        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
                        (raw['cause_name'] == cause) &  # cause
                        (raw['sex_name'] == 'Both') &  # only 'Both'
                        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')]  # DALYs
    dalys_rate_df = dalys_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    dalys_merged = dalys_number_df.merge(
        dalys_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    dalys_merged['population'] = dalys_merged['val_num'] / (dalys_merged['val_rate'] / 1e5)

    # Location ASDR
    asdr_results = run_location_asdr(
        dalys_merged,
        years=years,
        std_pop=std_pop
    )

    # save
    (output_dir / 'Location ASDR').mkdir(parents=True, exist_ok=True)
    for location in asdr_results['Location'].unique():
        LOGGER.info(f"Saving results for {location}")
        asdr_results[asdr_results['Location'] == location].to_csv(
            output_dir / 'Location ASDR' / f'{location}.csv',
            index=False
        )
    LOGGER.info(f"Results saved to {output_dir / 'Location ASDR'}")


if __name__ == '__main__':
    import os

    os.chdir(pathlib.Path(__file__).parent.parent)

    df: pd.DataFrame = joblib.load(pathlib.Path("data") / "data_cause_of_death_or_injury.pkl")

    run(
        df,
        std_pop=pd.read_csv(pathlib.Path("data") / 'standard_population.csv'),
        name='AnxietyDisorders.Statistics&YearTrend&LocationASDR.1990-2019',
        years=(1990, 2019)
    )
