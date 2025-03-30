from __future__ import annotations

import joblib
import pandas as pd
import pathlib
import typing
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


def run_asir_stat(
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
            incidence_df_ = incidence_df[(incidence_df['sex_name'] == 'Both') & (incidence_df['location_name'] ==
                                                                                 'Global')]
            # Calculate total cases
            incidence_cases_year = incidence_df_.copy().groupby('year').agg(_const.AGG_TEMPLATE).reset_index()
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
            incidence_merged_ = incidence_df_.copy()
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
            # Filter data based on specification type to ensure consistency with global calculation
            if spec == 'sex_name':
                # For sex-specific, only include Global location
                filtered_df = incidence_df[incidence_df['location_name'] == 'Global'].copy()
            else:  # location_name
                # For location-specific, only include Both sexes
                filtered_df = incidence_df[incidence_df['sex_name'] == 'Both'].copy()

            incidence_cases_year = filtered_df.groupby(['year', spec]).agg(_const.AGG_TEMPLATE).reset_index()
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
                incidence_merged_ = filtered_df[filtered_df[spec] == item].copy()
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


def run_asdr_stat(
    daly_df: pd.DataFrame,
    *,
    years: typing.Tuple[int, int],
    std_pop: pd.DataFrame,
    cause: str
) -> pd.DataFrame:
    """ASDR Statistics."""
    y1, y2 = years
    LOGGER.info(f"Calculating ASDR for {cause} from {y1} to {y2}")
    stat_results = pd.DataFrame(
        columns=[
            'Characteristic',
            f'ASDR per 1e5 population in {y1} (95% UI)',
            f'ASDR per 1e5 population in {y2} (95% UI)',
            'Percentage Change of ASDR (%, 95% UI)'
        ]
    )

    for spec in ['global', 'sex_name', 'location_name']:
        # Global
        if spec == 'global':
            daly_df_ = daly_df[
                (daly_df['sex_name'] == 'Both') &
                (daly_df['location_name'] == 'Global')
                ]
            # Aggregate DALYs by year
            daly_year = daly_df_.copy().groupby('year').agg(_const.AGG_TEMPLATE).reset_index()
            daly_year['val_num'] /= 1e5  # ASDR per 1e5
            daly_year['upper_num'] /= 1e5
            daly_year['lower_num'] /= 1e5

            LOGGER.info("=== Global ===")
            LOGGER.info(
                f"ASDR per 1e5 population in {y1}: "
                f"{daly_year[daly_year['year'] == y1]['val_num'].iloc[0]:.2f}"
            )
            LOGGER.info(
                f"95% UI: ({daly_year[daly_year['year'] == y1]['lower_num'].iloc[0]:.2f}, "
                f"{daly_year[daly_year['year'] == y1]['upper_num'].iloc[0]:.2f})"
            )
            LOGGER.info(
                f"ASDR per 1e5 population in {y2}: "
                f"{daly_year[daly_year['year'] == y2]['val_num'].iloc[0]:.2f}"
            )
            LOGGER.info(
                f"95% UI: ({daly_year[daly_year['year'] == y2]['lower_num'].iloc[0]:.2f}, "
                f"{daly_year[daly_year['year'] == y2]['upper_num'].iloc[0]:.2f})"
            )

            # Calculate percent change
            daly_y1 = daly_year[daly_year['year'] == y1].iloc[0]
            daly_y2 = daly_year[daly_year['year'] == y2].iloc[0]
            percent_change_val = calc_percent_change(daly_y1['val_num'], daly_y2['val_num'])
            percent_change_upper = calc_percent_change(daly_y1['upper_num'], daly_y2['upper_num'])
            percent_change_lower = calc_percent_change(daly_y1['lower_num'], daly_y2['lower_num'])

            LOGGER.info(f"Percentage change in ASDR from {y1} to {y2}: {percent_change_val:.2f}%")
            LOGGER.info(f"95% UI: ({percent_change_lower:.2f}%, {percent_change_upper:.2f}%)")

            # Calculate ASDR (per 1e5 population)
            daly_merged_ = daly_df_.copy()
            cases_age_y1 = daly_merged_[daly_merged_['year'] == y1].groupby('age_name').agg(
                _const.AGG_TEMPLATE
            ).reset_index()
            cases_age_y2 = daly_merged_[daly_merged_['year'] == y2].groupby('age_name').agg(
                _const.AGG_TEMPLATE
            ).reset_index()
            asdr1, asdr_lower1, asdr_upper1 = calc_asr(cases_age_y1, std_pop=std_pop, per=1e5)
            asdr2, asdr_lower2, asdr_upper2 = calc_asr(cases_age_y2, std_pop=std_pop, per=1e5)

            LOGGER.info(f"ASDR per 1e5 population in {y1}: {asdr1:.2f}")
            LOGGER.info(f"95% UI: ({asdr_lower1:.2f}, {asdr_upper1:.2f})")
            LOGGER.info(f"ASDR per 1e5 population in {y2}: {asdr2:.2f}")
            LOGGER.info(f"95% UI: ({asdr_lower2:.2f}, {asdr_upper2:.2f})")

            # Calculate percent change in ASDR
            percent_change_asdr = calc_percent_change(asdr1, asdr2)
            percent_change_asdr_lower = calc_percent_change(asdr_lower1, asdr_lower2)
            percent_change_asdr_upper = calc_percent_change(asdr_upper1, asdr_upper2)

            LOGGER.info(f"Percentage change in ASDR from {y1} to {y2}: {percent_change_asdr:.2f}%")
            LOGGER.info(f"95% UI: ({percent_change_asdr_lower:.2f}%, {percent_change_asdr_upper:.2f}%)")

            stat_results = pd.concat(
                [stat_results, pd.DataFrame(
                    {
                        'Characteristic':                            'Global',
                        f'ASDR per 1e5 population in {y1} (95% UI)': f"{daly_y1['val_num']:.2f} ("
                                                                     f"{daly_y1['lower_num']:.2f}, "
                                                                     f"{daly_y1['upper_num']:.2f})",
                        f'ASDR per 1e5 population in {y2} (95% UI)': f"{daly_y2['val_num']:.2f} ("
                                                                     f"{daly_y2['lower_num']:.2f}, "
                                                                     f"{daly_y2['upper_num']:.2f})",
                        'Percentage Change of ASDR (%, 95% UI)':     f"{percent_change_val:.2f} ("
                                                                     f"{percent_change_lower:.2f}, "
                                                                     f"{percent_change_upper:.2f})"
                    },
                    index=[0]
                )],
                ignore_index=True
            )
        # Sex-specific and Location-specific
        else:
            daly_cases_year = daly_df.copy().groupby(['year', spec]).agg(_const.AGG_TEMPLATE).reset_index()
            daly_cases_year['val_num'] /= 1e5
            daly_cases_year['upper_num'] /= 1e5
            daly_cases_year['lower_num'] /= 1e5

            # Each item of the characteristic
            for item in daly_cases_year[spec].unique():
                cases_spec = daly_cases_year[daly_cases_year[spec] == item]
                try:
                    daly_y1 = cases_spec[cases_spec['year'] == y1].iloc[0]
                    daly_y2 = cases_spec[cases_spec['year'] == y2].iloc[0]
                except IndexError:
                    LOGGER.warning(f"No data for {spec} - {item} in year {y1} or {y2}. Skipping.")
                    continue

                percent_change_val = calc_percent_change(daly_y1['val_num'], daly_y2['val_num'])
                percent_change_upper = calc_percent_change(
                    daly_y1['upper_num'],
                    daly_y2['upper_num']
                )
                percent_change_lower = calc_percent_change(
                    daly_y1['lower_num'],
                    daly_y2['lower_num']
                )

                LOGGER.info(f"=== {spec.capitalize()} - {item} ===")
                LOGGER.info(f"ASDR per 1e5 population in {y1}: {daly_y1['val_num']:.2f}")
                LOGGER.info(
                    f"95% UI: ({daly_y1['lower_num']:.2f}, {daly_y1['upper_num']:.2f})"
                )
                LOGGER.info(f"ASDR per 1e5 population in {y2}: {daly_y2['val_num']:.2f}")
                LOGGER.info(
                    f"95% UI: ({daly_y2['lower_num']:.2f}, {daly_y2['upper_num']:.2f})"
                )
                LOGGER.info(
                    f"Percentage change in ASDR from {y1} to {y2} of {spec.capitalize()}: {percent_change_val:.2f}%"
                )
                LOGGER.info(f"95% UI: ({percent_change_lower:.2f}%, {percent_change_upper:.2f}%)")

                # Calculate ASDR (per 1e5 population)
                daly_merged_ = daly_df.copy()[daly_df[spec] == item]
                cases_age_y1 = daly_merged_[daly_merged_['year'] == y1].groupby('age_name').agg(
                    _const.AGG_TEMPLATE
                ).reset_index()
                cases_age_y2 = daly_merged_[daly_merged_['year'] == y2].groupby('age_name').agg(
                    _const.AGG_TEMPLATE
                ).reset_index()
                asdr1, asdr_lower1, asdr_upper1 = calc_asr(cases_age_y1, std_pop=std_pop, per=1e5)
                asdr2, asdr_lower2, asdr_upper2 = calc_asr(cases_age_y2, std_pop=std_pop, per=1e5)

                LOGGER.info(f"ASDR per 1e5 population in {y1}: {asdr1:.2f}")
                LOGGER.info(f"95% UI: ({asdr_lower1:.2f}, {asdr_upper1:.2f})")
                LOGGER.info(f"ASDR per 1e5 population in {y2}: {asdr2:.2f}")
                LOGGER.info(f"95% UI: ({asdr_lower2:.2f}, {asdr_upper2:.2f})")

                # Calculate percent change in ASDR
                percent_change_asdr = calc_percent_change(asdr1, asdr2)
                percent_change_asdr_lower = calc_percent_change(asdr_lower1, asdr_lower2)
                percent_change_asdr_upper = calc_percent_change(asdr_upper1, asdr_upper2)

                LOGGER.info(f"Percentage change in ASDR from {y1} to {y2}: {percent_change_asdr:.2f}%")
                LOGGER.info(f"95% UI: ({percent_change_asdr_lower:.2f}%, {percent_change_asdr_upper:.2f}%)")

                stat_results = pd.concat(
                    [stat_results, pd.DataFrame(
                        {
                            'Characteristic':                            f"{spec.capitalize()} - {item}",
                            f'ASDR per 1e5 population in {y1} (95% UI)': f"{daly_y1['val_num']:.2f} ("
                                                                         f"{daly_y1['lower_num']:.2f}, "
                                                                         f"{daly_y1['upper_num']:.2f})",
                            f'ASDR per 1e5 population in {y2} (95% UI)': f"{daly_y2['val_num']:.2f} ("
                                                                         f"{daly_y2['lower_num']:.2f}, "
                                                                         f"{daly_y2['upper_num']:.2f})",
                            'Percentage Change of ASDR (%, 95% UI)':     f"{percent_change_val:.2f} ("
                                                                         f"{percent_change_lower:.2f}, "
                                                                         f"{percent_change_upper:.2f})"
                        },
                        index=[0]
                    )],
                    ignore_index=True
                )

    LOGGER.info("=== ASDR Statistics Results ===")
    print(tabulate(stat_results, headers='keys', tablefmt='pretty'))
    return stat_results


def run_aspr_stat(
    prevalence_df: pd.DataFrame,
    *,
    years: typing.Tuple[int, int],
    std_pop: pd.DataFrame,
    cause: str
) -> pd.DataFrame:
    """ASPR Statistics."""
    y1, y2 = years
    LOGGER.info(f"Calculating ASPR for {cause} from {y1} to {y2}")
    stat_results = pd.DataFrame(
        columns=[
            'Characteristic',
            f'ASPR per 1e5 population in {y1} (95% UI)',
            f'ASPR per 1e5 population in {y2} (95% UI)',
            'Percentage Change of ASPR (%, 95% UI)'
        ]
    )

    for spec in ['global', 'sex_name', 'location_name']:
        # Global
        if spec == 'global':
            prevalence_df_ = prevalence_df[
                (prevalence_df['sex_name'] == 'Both') &
                (prevalence_df['location_name'] == 'Global')
                ]
            # Aggregate Prevalence by year
            prevalence_year = prevalence_df_.copy().groupby('year').agg(_const.AGG_TEMPLATE).reset_index()
            prevalence_year['val_num'] /= 1e5  # ASPR per 1e5
            prevalence_year['upper_num'] /= 1e5
            prevalence_year['lower_num'] /= 1e5

            LOGGER.info("=== Global ===")
            LOGGER.info(
                f"ASPR per 1e5 population in {y1}: "
                f"{prevalence_year[prevalence_year['year'] == y1]['val_num'].iloc[0]:.2f}"
            )
            LOGGER.info(
                f"95% UI: ({prevalence_year[prevalence_year['year'] == y1]['lower_num'].iloc[0]:.2f}, "
                f"{prevalence_year[prevalence_year['year'] == y1]['upper_num'].iloc[0]:.2f})"
            )
            LOGGER.info(
                f"ASPR per 1e5 population in {y2}: "
                f"{prevalence_year[prevalence_year['year'] == y2]['val_num'].iloc[0]:.2f}"
            )
            LOGGER.info(
                f"95% UI: ({prevalence_year[prevalence_year['year'] == y2]['lower_num'].iloc[0]:.2f}, "
                f"{prevalence_year[prevalence_year['year'] == y2]['upper_num'].iloc[0]:.2f})"
            )

            # Calculate percent change
            prevalence_y1 = prevalence_year[prevalence_year['year'] == y1].iloc[0]
            prevalence_y2 = prevalence_year[prevalence_year['year'] == y2].iloc[0]
            percent_change_val = calc_percent_change(prevalence_y1['val_num'], prevalence_y2['val_num'])
            percent_change_upper = calc_percent_change(prevalence_y1['upper_num'], prevalence_y2['upper_num'])
            percent_change_lower = calc_percent_change(prevalence_y1['lower_num'], prevalence_y2['lower_num'])

            LOGGER.info(f"Percentage change in ASPR from {y1} to {y2}: {percent_change_val:.2f}%")
            LOGGER.info(f"95% UI: ({percent_change_lower:.2f}%, {percent_change_upper:.2f}%)")

            # Calculate ASPR (per 1e5 population)
            prevalence_merged_ = prevalence_df_.copy()
            cases_age_y1 = prevalence_merged_[prevalence_merged_['year'] == y1].groupby('age_name').agg(
                _const.AGG_TEMPLATE
            ).reset_index()
            cases_age_y2 = prevalence_merged_[prevalence_merged_['year'] == y2].groupby('age_name').agg(
                _const.AGG_TEMPLATE
            ).reset_index()
            aspr1, aspr_lower1, aspr_upper1 = calc_asr(cases_age_y1, std_pop=std_pop, per=1e5)
            aspr2, aspr_lower2, aspr_upper2 = calc_asr(cases_age_y2, std_pop=std_pop, per=1e5)

            LOGGER.info(f"ASPR per 1e5 population in {y1}: {aspr1:.2f}")
            LOGGER.info(f"95% UI: ({aspr_lower1:.2f}, {aspr_upper1:.2f})")
            LOGGER.info(f"ASPR per 1e5 population in {y2}: {aspr2:.2f}")
            LOGGER.info(f"95% UI: ({aspr_lower2:.2f}, {aspr_upper2:.2f})")

            # Calculate percent change in ASPR
            percent_change_aspr = calc_percent_change(aspr1, aspr2)
            percent_change_aspr_lower = calc_percent_change(aspr_lower1, aspr_lower2)
            percent_change_aspr_upper = calc_percent_change(aspr_upper1, aspr_upper2)

            LOGGER.info(f"Percentage change in ASPR from {y1} to {y2}: {percent_change_aspr:.2f}%")
            LOGGER.info(f"95% UI: ({percent_change_aspr_lower:.2f}%, {percent_change_aspr_upper:.2f}%)")

            stat_results = pd.concat(
                [stat_results, pd.DataFrame(
                    {
                        'Characteristic':                            'Global',
                        f'ASPR per 1e5 population in {y1} (95% UI)': f"{prevalence_y1['val_num']:.2f} ("
                                                                     f"{prevalence_y1['lower_num']:.2f}, "
                                                                     f"{prevalence_y1['upper_num']:.2f})",
                        f'ASPR per 1e5 population in {y2} (95% UI)': f"{prevalence_y2['val_num']:.2f} ("
                                                                     f"{prevalence_y2['lower_num']:.2f}, "
                                                                     f"{prevalence_y2['upper_num']:.2f})",
                        'Percentage Change of ASPR (%, 95% UI)':     f"{percent_change_val:.2f} ("
                                                                     f"{percent_change_lower:.2f}, "
                                                                     f"{percent_change_upper:.2f})"
                    },
                    index=[0]
                )],
                ignore_index=True
            )
        # Sex-specific and Location-specific
        else:
            prevalence_cases_year = prevalence_df.copy().groupby(['year', spec]).agg(_const.AGG_TEMPLATE).reset_index()
            prevalence_cases_year['val_num'] /= 1e5
            prevalence_cases_year['upper_num'] /= 1e5
            prevalence_cases_year['lower_num'] /= 1e5

            # Each item of the characteristic
            for item in prevalence_cases_year[spec].unique():
                cases_spec = prevalence_cases_year[prevalence_cases_year[spec] == item]
                try:
                    prevalence_y1 = cases_spec[cases_spec['year'] == y1].iloc[0]
                    prevalence_y2 = cases_spec[cases_spec['year'] == y2].iloc[0]
                except IndexError:
                    LOGGER.warning(f"No data for {spec} - {item} in year {y1} or {y2}. Skipping.")
                    continue

                percent_change_val = calc_percent_change(prevalence_y1['val_num'], prevalence_y2['val_num'])
                percent_change_upper = calc_percent_change(
                    prevalence_y1['upper_num'],
                    prevalence_y2['upper_num']
                )
                percent_change_lower = calc_percent_change(
                    prevalence_y1['lower_num'],
                    prevalence_y2['lower_num']
                )

                LOGGER.info(f"=== {spec.capitalize()} - {item} ===")
                LOGGER.info(f"ASPR per 1e5 population in {y1}: {prevalence_y1['val_num']:.2f}")
                LOGGER.info(
                    f"95% UI: ({prevalence_y1['lower_num']:.2f}, {prevalence_y1['upper_num']:.2f})"
                )
                LOGGER.info(f"ASPR per 1e5 population in {y2}: {prevalence_y2['val_num']:.2f}")
                LOGGER.info(
                    f"95% UI: ({prevalence_y2['lower_num']:.2f}, {prevalence_y2['upper_num']:.2f})"
                )
                LOGGER.info(
                    f"Percentage change in ASPR from {y1} to {y2} of {spec.capitalize()}: {percent_change_val:.2f}%"
                )
                LOGGER.info(f"95% UI: ({percent_change_lower:.2f}%, {percent_change_upper:.2f}%)")

                # Calculate ASPR (per 1e5 population)
                prevalence_merged_ = prevalence_df.copy()[prevalence_df[spec] == item]
                cases_age_y1 = prevalence_merged_[prevalence_merged_['year'] == y1].groupby('age_name').agg(
                    _const.AGG_TEMPLATE
                ).reset_index()
                cases_age_y2 = prevalence_merged_[prevalence_merged_['year'] == y2].groupby('age_name').agg(
                    _const.AGG_TEMPLATE
                ).reset_index()
                aspr1, aspr_lower1, aspr_upper1 = calc_asr(cases_age_y1, std_pop=std_pop, per=1e5)
                aspr2, aspr_lower2, aspr_upper2 = calc_asr(cases_age_y2, std_pop=std_pop, per=1e5)

                LOGGER.info(f"ASPR per 1e5 population in {y1}: {aspr1:.2f}")
                LOGGER.info(f"95% UI: ({aspr_lower1:.2f}, {aspr_upper1:.2f})")
                LOGGER.info(f"ASPR per 1e5 population in {y2}: {aspr2:.2f}")
                LOGGER.info(f"95% UI: ({aspr_lower2:.2f}, {aspr_upper2:.2f})")

                # Calculate percent change in ASPR
                percent_change_aspr = calc_percent_change(aspr1, aspr2)
                percent_change_aspr_lower = calc_percent_change(aspr_lower1, aspr_lower2)
                percent_change_aspr_upper = calc_percent_change(aspr_upper1, aspr_upper2)

                LOGGER.info(f"Percentage change in ASPR from {y1} to {y2}: {percent_change_aspr:.2f}%")
                LOGGER.info(f"95% UI: ({percent_change_aspr_lower:.2f}%, {percent_change_aspr_upper:.2f}%)")

                stat_results = pd.concat(
                    [stat_results, pd.DataFrame(
                        {
                            'Characteristic':                            f"{spec.capitalize()} - {item}",
                            f'ASPR per 1e5 population in {y1} (95% UI)': f"{prevalence_y1['val_num']:.2f} ("
                                                                         f"{prevalence_y1['lower_num']:.2f}, "
                                                                         f"{prevalence_y1['upper_num']:.2f})",
                            f'ASPR per 1e5 population in {y2} (95% UI)': f"{prevalence_y2['val_num']:.2f} ("
                                                                         f"{prevalence_y2['lower_num']:.2f}, "
                                                                         f"{prevalence_y2['upper_num']:.2f})",
                            'Percentage Change of ASPR (%, 95% UI)':     f"{percent_change_val:.2f} ("
                                                                         f"{percent_change_lower:.2f}, "
                                                                         f"{percent_change_upper:.2f})"
                        },
                        index=[0]
                    )],
                    ignore_index=True
                )

    LOGGER.info("=== ASPR Statistics Results ===")
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
    incidence_number_df = raw[
        (raw['metric_name'] == 'Number') &  # number of cases
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['measure_name'] == 'Incidence')  # incidence
        ]
    incidence_number_df = incidence_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    # rate
    incidence_rate_df = raw[
        (raw['metric_name'] == 'Rate') &  # rate of cases per 1e6
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['measure_name'] == 'Incidence')  # incidence
        ]
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
    asir_stat_results = run_asir_stat(
        incidence_merged,
        years=years,
        std_pop=std_pop,
        cause=cause
    )
    # save ASIR results
    (output_dir / f'Number and ASIR of {cause}').mkdir(parents=True, exist_ok=True)
    asir_stat_results.to_csv(output_dir / f'Number and ASIR of {cause}' / 'result.csv', index=False)
    LOGGER.info(f"ASIR Results saved to {output_dir / f'Number and ASIR of {cause}' / 'result.csv'}")

    # ASDR Statistics ---------------------------------------------------------
    # DALYs Data ------------------------------------------------------------
    dalys_number_df = raw[
        (raw['metric_name'] == 'Number') &  # number of DALYs
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')  # DALYs
        ]
    dalys_number_df = dalys_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    dalys_rate_df = raw[
        (raw['metric_name'] == 'Rate') &  # rate of DALYs per 1e5
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')  # DALYs
        ]
    dalys_rate_df = dalys_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    dalys_merged = dalys_number_df.merge(
        dalys_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    dalys_merged['population'] = dalys_merged['val_num'] / (dalys_merged['val_rate'] / 1e5)

    # Run ASDR Statistics
    asdr_stat_results = run_asdr_stat(
        daly_df=dalys_merged,
        years=years,
        std_pop=std_pop,
        cause=cause
    )
    # save ASDR results
    (output_dir / f'Number and ASDR of {cause}').mkdir(parents=True, exist_ok=True)
    asdr_stat_results.to_csv(output_dir / f'Number and ASDR of {cause}' / 'asdr_result.csv', index=False)
    LOGGER.info(f"ASDR Results saved to {output_dir / f'Number and ASDR of {cause}' / 'asdr_result.csv'}")

    # ASPR Statistics ---------------------------------------------------------
    # Prevalence Data ---------------------------------------------------------
    prevalence_number_df = raw[
        (raw['metric_name'] == 'Number') &  # number of cases
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'Prevalence')  # prevalence
        ]
    prevalence_number_df = prevalence_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    prevalence_rate_df = raw[
        (raw['metric_name'] == 'Rate') &  # rate of cases per 1e5
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'Prevalence')  # prevalence
        ]
    prevalence_rate_df = prevalence_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    prevalence_merged = prevalence_number_df.merge(
        prevalence_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    prevalence_merged['population'] = prevalence_merged['val_num'] / (prevalence_merged['val_rate'] / 1e5)

    # Run ASPR Statistics
    aspr_stat_results = run_aspr_stat(
        prevalence_df=prevalence_merged,
        years=years,
        std_pop=std_pop,
        cause=cause
    )
    # save ASPR results
    (output_dir / f'Number and ASPR of {cause}').mkdir(parents=True, exist_ok=True)
    aspr_stat_results.to_csv(output_dir / f'Number and ASPR of {cause}' / 'aspr_result.csv', index=False)
    LOGGER.info(f"ASPR Results saved to {output_dir / f'Number and ASPR of {cause}' / 'aspr_result.csv'}")

    # Year Trend -------------------------------------------------------------
    # Incidence Data (for Year Trend)
    incidence_yearly_number_df = raw[
        (raw['metric_name'] == 'Number') &  # number of cases
        (raw['location_name'] == 'Global') &  # global
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'Incidence')  # incidence
        ]
    incidence_yearly_number_df = incidence_yearly_number_df.groupby(SPEC_COLUMNS).agg(
        _const.NUMBER_AGG_TEMPLATE
    ).reset_index()

    incidence_yearly_rate_df = raw[
        (raw['metric_name'] == 'Rate') &  # rate of cases per 1e5
        (raw['location_name'] == 'Global') &  # global
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'Incidence')  # incidence
        ]
    incidence_yearly_rate_df = incidence_yearly_rate_df.groupby(SPEC_COLUMNS).agg(
        _const.RATE_AGG_TEMPLATE
    ).reset_index()

    incidence_yearly_merged = incidence_yearly_number_df.merge(
        incidence_yearly_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    incidence_yearly_merged['population'] = incidence_yearly_merged['val_num'] / (
            incidence_yearly_merged['val_rate'] / 1e5)

    # DALYs Data (for Year Trend)
    dalys_yearly_number_df = raw[
        (raw['metric_name'] == 'Number') &  # number of DALYs
        (raw['location_name'] == 'Global') &  # global
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')  # DALYs
        ]
    dalys_yearly_number_df = dalys_yearly_number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    dalys_yearly_rate_df = raw[
        (raw['metric_name'] == 'Rate') &  # rate of DALYs per 1e5
        (raw['location_name'] == 'Global') &  # global
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')  # DALYs
        ]
    dalys_yearly_rate_df = dalys_yearly_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    dalys_yearly_merged = dalys_yearly_number_df.merge(
        dalys_yearly_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    dalys_yearly_merged['population'] = dalys_yearly_merged['val_num'] / (dalys_yearly_merged['val_rate'] / 1e5)

    # Prevalence Data (for Year Trend)
    prevalence_yearly_number_df = raw[
        (raw['metric_name'] == 'Number') &  # number of cases
        (raw['location_name'] == 'Global') &  # global
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'Prevalence')  # prevalence
        ]
    prevalence_yearly_number_df = prevalence_yearly_number_df.groupby(SPEC_COLUMNS).agg(
        _const.NUMBER_AGG_TEMPLATE
    ).reset_index()

    prevalence_yearly_rate_df = raw[
        (raw['metric_name'] == 'Rate') &  # rate of cases per 1e5
        (raw['location_name'] == 'Global') &  # global
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(range(years[0], years[1] + 1))) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'Prevalence')  # prevalence
        ]
    prevalence_yearly_rate_df = prevalence_yearly_rate_df.groupby(SPEC_COLUMNS).agg(
        _const.RATE_AGG_TEMPLATE
    ).reset_index()

    prevalence_yearly_merged = prevalence_yearly_number_df.merge(
        prevalence_yearly_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    prevalence_yearly_merged['population'] = prevalence_yearly_merged['val_num'] / (
            prevalence_yearly_merged['val_rate'] / 1e5)

    # Run Year Trend
    asir_results, asdr_results, aspr_results = run_year_trend(
        incidence_df=incidence_yearly_merged,
        daly_df=dalys_yearly_merged,
        prevalence_df=prevalence_yearly_merged,
        years=years,
        std_pop=std_pop,
        cause=cause
    )
    # save Year Trend results
    (output_dir / 'Year Trend' / 'ASIR').mkdir(parents=True, exist_ok=True)
    (output_dir / 'Year Trend' / 'ASDR').mkdir(parents=True, exist_ok=True)
    (output_dir / 'Year Trend' / 'ASPR').mkdir(parents=True, exist_ok=True)
    asir_results.to_csv(output_dir / 'Year Trend' / 'ASIR' / 'trend.asir.csv', index=False)
    asdr_results.to_csv(output_dir / 'Year Trend' / 'ASDR' / 'trend.asdr.csv', index=False)
    aspr_results.to_csv(output_dir / 'Year Trend' / 'ASPR' / 'trend.aspr.csv', index=False)
    LOGGER.info(f"Year Trend Results saved to {output_dir / 'Year Trend'}")

    # Location ASDR ----------------------------------------------------------
    # DALYs Data for Location ASDR
    dalys_location_number_df = raw[
        (raw['metric_name'] == 'Number') &  # number of DALYs
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')  # DALYs
        ]
    dalys_location_number_df = dalys_location_number_df.groupby(SPEC_COLUMNS).agg(
        _const.NUMBER_AGG_TEMPLATE
    ).reset_index()

    dalys_location_rate_df = raw[
        (raw['metric_name'] == 'Rate') &  # rate of DALYs per 1e5
        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
        (raw['year'].isin(years)) &  # years
        (raw['cause_name'] == cause) &  # cause
        (raw['sex_name'] == 'Both') &  # only 'Both'
        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')  # DALYs
        ]
    dalys_location_rate_df = dalys_location_rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    dalys_location_merged = dalys_location_number_df.merge(
        dalys_location_rate_df,
        on=SPEC_COLUMNS,
        suffixes=('_num', '_rate')
    )
    dalys_location_merged['population'] = dalys_location_merged['val_num'] / (dalys_location_merged['val_rate'] / 1e5)

    # Run Location ASDR Statistics
    asdr_location_results = run_location_asdr(
        daly_df=dalys_location_merged,
        years=years,
        std_pop=std_pop
    )
    # save Location ASDR results
    (output_dir / 'Location ASDR').mkdir(parents=True, exist_ok=True)
    for location in asdr_location_results['Location'].unique():
        LOGGER.info(f"Saving results for {location}")
        asdr_location_results[asdr_location_results['Location'] == location].to_csv(
            output_dir / 'Location ASDR' / f'{location}.csv',
            index=False
        )
    LOGGER.info(f"Location ASDR Results saved to {output_dir / 'Location ASDR'}")


if __name__ == '__main__':
    import os

    os.chdir(pathlib.Path(__file__).parent.parent)

    df: pd.DataFrame = joblib.load(pathlib.Path("data") / "data_cause_of_death_or_injury.pkl")

    run(
        df,
        std_pop=pd.read_csv(pathlib.Path("data") / 'standard_population.csv'),
        name='AnxietyDisorders.Statistics&YearTrend&LocationASDR.1992-2021',
        years=(1992, 2021)
    )
