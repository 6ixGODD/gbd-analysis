# Prepare data for Age-Period-Cohort analysis
# See https://analysistools.cancer.gov/apc/

import pathlib
import typing

import pandas as pd
from matplotlib import pyplot as plt

from gbd.common import (
    const as _const,
    utils as _utils,
)

plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'

LOGGER = _utils.Logger()

SPEC_COLUMNS = ['age_name', 'sex_name', 'year']


def _plot_line(
    raw: pd.DataFrame,
    *,
    y_col: str,
    x_col: str = 'Age',
    up_col: str = 'CI Hi',
    low_col: str = 'CI Lo',
    both_hor: typing.Optional[float] = None,
    both_hor_lo: typing.Optional[float] = None,
    both_hor_hi: typing.Optional[float] = None,
    female_hor: typing.Optional[float] = None,
    female_hor_lo: typing.Optional[float] = None,
    female_hor_hi: typing.Optional[float] = None,
    male_hor: typing.Optional[float] = None,
    male_hor_lo: typing.Optional[float] = None,
    male_hor_hi: typing.Optional[float] = None,
    y_label: str,
    x_label: str = 'Age',
    title: str,
    output_dir: pathlib.Path,
):
    b_df = raw[raw['Sex'] == 'Both']
    f_df = raw[raw['Sex'] == 'Female']
    m_df = raw[raw['Sex'] == 'Male']

    plt.figure(figsize=(12, 8))
    plt.plot(b_df[x_col], b_df[y_col], label='Both', marker='o', ms=4, color='blue')
    plt.fill_between(
        b_df[x_col],
        b_df[low_col],
        b_df[up_col],
        alpha=0.2,
        color='blue',
        edgecolor='none'
    )
    if both_hor:
        # Horizontal line
        plt.axhline(y=both_hor, color='blue', linestyle='--')
    if both_hor_lo and both_hor_hi:
        plt.fill_between(
            b_df[x_col],
            both_hor_lo,
            both_hor_hi,
            alpha=0.2,
            color='blue',
            edgecolor='none'
        )

    plt.plot(f_df[x_col], f_df[y_col], label='Female', marker='^', ms=4, color='red')
    plt.fill_between(
        f_df[x_col],
        f_df[low_col],
        f_df[up_col],
        alpha=0.2,
        color='red',
        edgecolor='none'
    )
    if female_hor:
        plt.axhline(y=female_hor, color='red', linestyle='--')
    if female_hor_lo and female_hor_hi:
        plt.fill_between(
            f_df[x_col],
            female_hor_lo,
            female_hor_hi,
            alpha=0.2,
            color='red',
            edgecolor='none'
        )

    plt.plot(m_df[x_col], m_df[y_col], label='Male', marker='s', ms=4, color='green')
    plt.fill_between(
        m_df[x_col],
        m_df[low_col],
        m_df[up_col],
        alpha=0.2,
        color='green',
        edgecolor='none'
    )
    if male_hor:
        plt.axhline(y=male_hor, color='green', linestyle='--')
    if male_hor_lo and male_hor_hi:
        plt.fill_between(
            m_df[x_col],
            male_hor_lo,
            male_hor_hi,
            alpha=0.2,
            color='green',
            edgecolor='none'
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / f'{title}.png')
    LOGGER.info(f"Saved to {output_dir / f'{title}.png'}")
    plt.close()


def plot_apc(
    *,
    age_deviations: pd.DataFrame,
    period_deviations: pd.DataFrame,
    cohort_deviations: pd.DataFrame,
    longitudinal_age_curve: pd.DataFrame,
    cross_sectional_age_curve: pd.DataFrame,
    long_vs_cross_rr: pd.DataFrame,
    fitted_temporal_trends: pd.DataFrame,
    period_rr: pd.DataFrame,
    cohort_rr: pd.DataFrame,
    local_drifts: pd.DataFrame,
    net_drifts: pd.DataFrame,
    output_dir: pathlib.Path,
):

    _plot_line(
        age_deviations,
        y_col='Deviation',
        y_label='Deviation',
        title='Age Deviations',
        output_dir=output_dir,
    )

    _plot_line(
        period_deviations,
        x_col='Period',
        y_col='Deviation',
        x_label='Period',
        y_label='Deviation',
        title='Period Deviations',
        output_dir=output_dir,
    )
    _plot_line(
        cohort_deviations,
        x_col='Cohort',
        y_col='Deviation',
        x_label='Cohort',
        y_label='Deviation',
        title='Cohort Deviations',
        output_dir=output_dir,
    )
    _plot_line(
        longitudinal_age_curve,
        y_col='Rate',
        y_label='Rate',
        up_col='CIHi',
        low_col='CILo',
        title='Longitudinal Age Curve',
        output_dir=output_dir,
    )
    _plot_line(
        cross_sectional_age_curve,
        y_col='Rate',
        y_label='Rate',
        up_col='CIHi',
        low_col='CILo',
        title='Cross Sectional Age Curve',
        output_dir=output_dir,
    )
    _plot_line(
        long_vs_cross_rr,
        y_col='Rate Ratio',
        y_label='Rate Ratio',
        up_col='CIHi',
        low_col='CILo',
        title='Longitudinal vs Cross Sectional Rate Ratio',
        output_dir=output_dir,
    )
    _plot_line(
        fitted_temporal_trends,
        x_col='Period',
        y_col='Rate',
        x_label='Period',
        y_label='Rate',
        up_col='CIHi',
        low_col='CILo',
        title='Fitted Temporal Trends',
        output_dir=output_dir,
    )
    _plot_line(
        period_rr,
        x_col='Period',
        y_col='Rate Ratio',
        x_label='Period',
        y_label='Rate Ratio',
        up_col='CIHi',
        low_col='CILo',
        title='Period Rate Ratio',
        output_dir=output_dir,
    )
    _plot_line(
        cohort_rr,
        x_col='Cohort',
        y_col='Rate Ratio',
        x_label='Cohort',
        y_label='Rate Ratio',
        up_col='CIHi',
        low_col='CILo',
        title='Cohort Rate Ratio',
        output_dir=output_dir,
    )

    both_net_drift = net_drifts[net_drifts['Sex'] == 'Both']
    female_net_drift = net_drifts[net_drifts['Sex'] == 'Female']
    male_net_drift = net_drifts[net_drifts['Sex'] == 'Male']

    _plot_line(
        local_drifts,
        y_col='Percent per Year',
        y_label='Percent per Year',
        up_col='CIHi',
        low_col='CILo',
        both_hor=both_net_drift['Net Drift (%/year)'].values[0],
        both_hor_lo=both_net_drift['CI Lo'].values[0],
        both_hor_hi=both_net_drift['CI Hi'].values[0],
        female_hor=female_net_drift['Net Drift (%/year)'].values[0],
        female_hor_lo=female_net_drift['CI Lo'].values[0],
        female_hor_hi=female_net_drift['CI Hi'].values[0],
        male_hor=male_net_drift['Net Drift (%/year)'].values[0],
        male_hor_lo=male_net_drift['CI Lo'].values[0],
        male_hor_hi=male_net_drift['CI Hi'].values[0],
        title='Local Drifts with Net Drifts',
        output_dir=output_dir,
    )


def apc_data(
    raw: pd.DataFrame,
    *,
    title: str = "Anxiety Disorders",
    description: str = "Anxiety Disorders",
    years: typing.Tuple[int, int] = (1990, 2019),
    interval: int = 5,
    output_dir: pathlib.Path = pathlib.Path("output"),
    cause: str = 'Anxiety disorders',
):
    # Process Data -----------------------------------------------------------
    number_df = raw[(raw['metric_name'] == 'Number') &  # number of cases
                    (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                    (raw['cause_name'] == cause) &  # cause
                    (raw['location_name'] == 'Global') &  # global
                    (raw['measure_name'] == 'Incidence')]  # incidence
    number_df = number_df.groupby(SPEC_COLUMNS).agg(_const.NUMBER_AGG_TEMPLATE).reset_index()

    rate_df = raw[(raw['metric_name'] == 'Rate') &  # rate of cases
                  (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                  (raw['cause_name'] == cause) &  # cause
                  (raw['location_name'] == 'Global') &  # global
                  (raw['measure_name'] == 'Incidence')]  # incidence
    rate_df = rate_df.groupby(SPEC_COLUMNS).agg(_const.RATE_AGG_TEMPLATE).reset_index()

    merged_df = pd.merge(number_df, rate_df, on=SPEC_COLUMNS, suffixes=('_num', '_rate'))
    merged_df['population'] = merged_df['val_num'] / (merged_df['val_rate'] / 1e5)

    year_group = [(y, y + interval) for y in range(years[0], years[1], interval)]
    LOGGER.info(f"Year Group: {year_group}")
    for gender in ['Male', 'Female', 'Both']:
        LOGGER.info(f"Processing {gender} ...")
        gender_df = merged_df[merged_df['sex_name'] == gender]
        gender_result = []
        for age_group in _const.AGE_GROUPS:
            age_result = []
            for start, end in year_group:
                sub_df = gender_df[(gender_df['age_name'] == age_group) &
                                   (gender_df['year'] >= start) & (gender_df['year'] < end)]
                if sub_df.empty:
                    continue
                sub_df = sub_df.groupby('year').agg({'val_num': 'sum', 'population': 'sum'}).reset_index()
                age_result.extend([sub_df['val_num'].sum(), sub_df['population'].sum()])
            gender_result.append(age_result)
        gender_result_df = pd.DataFrame(gender_result)
        # Add rows at the front of the DataFrame
        meta_df = pd.DataFrame(
            [
                [f'Title: {title} - Gender: {gender}'] + [''] * (len(gender_result_df.columns) - 1),
                [f'Description: {description}'] + [''] * (len(gender_result_df.columns) - 1),
                [f'Start Year: {years[0]}'] + [''] * (len(gender_result_df.columns) - 1),
                ['Start Age: 0'] + [''] * (len(gender_result_df.columns) - 1),
                [f'Interval (Years): {interval}'] + [''] * (len(gender_result_df.columns) - 1),
            ]
        )
        gender_result_df = pd.concat([meta_df, gender_result_df], ignore_index=True)
        gender_result_df.to_csv(
            output_dir / f'apc.{gender}.csv',
            index=False,
            header=False,
            quoting=0,
            quotechar='"'
        )
        LOGGER.info(f"Saved to {output_dir / f'apc.{gender}.csv'}")


def run(
    *,
    raw: typing.Optional[pd.DataFrame] = None,
    title: typing.Optional[str] = None,
    description: typing.Optional[str] = None,
    cause: typing.Optional[str] = None,
    interval: typing.Optional[int] = None,
    years: typing.Optional[typing.Tuple[int, int]] = None,

    both_xlsx_path: typing.Optional[pathlib.Path] = None,
    f_xlsx_path: typing.Optional[pathlib.Path] = None,
    m_xlsx_path: typing.Optional[pathlib.Path] = None,

    output_dir: pathlib.Path = pathlib.Path("output"),
    name: str = 'exp',
):
    # Initialize -------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _utils.unique_path(output_dir, name)

    if raw:
        LOGGER.info(f"Processing APC data ...")

        apc_data(
            raw,
            title=title or "Anxiety Disorders",
            description=description or "Anxiety Disorders",
            cause=cause or 'Anxiety disorders',
            interval=interval or 5,
            years=years or (1990, 2019),
            output_dir=output_dir,
        )
    else:
        LOGGER.info(f"Plotting APC data from {both_xlsx_path}, {f_xlsx_path}, {m_xlsx_path} ...")

        both_age_deviations = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='AgeDeviations')
        both_age_deviations['Sex'] = 'Both'
        both_period_deviations = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='PerDeviations')
        both_period_deviations['Sex'] = 'Both'
        both_cohort_deviations = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='CohDeviations')
        both_cohort_deviations['Sex'] = 'Both'
        both_longitudinal_age_curve = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='LongAge')
        both_longitudinal_age_curve['Sex'] = 'Both'
        both_cross_sectional_age_curve = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='CrossAge')
        both_cross_sectional_age_curve['Sex'] = 'Both'
        both_long_vs_cross_rr = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='Long2CrossRR')
        both_long_vs_cross_rr['Sex'] = 'Both'
        both_fitted_temporal_trends = pd.read_excel(
            both_xlsx_path,
            engine="openpyxl",
            sheet_name='FittedTemporalTrends'
        )
        both_fitted_temporal_trends['Sex'] = 'Both'
        both_period_rr = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='PeriodRR')
        both_period_rr['Sex'] = 'Both'
        both_cohort_rr = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='CohortRR')
        both_cohort_rr['Sex'] = 'Both'
        both_local_drifts = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='LocalDrifts')
        both_local_drifts['Sex'] = 'Both'
        both_net_drifts = pd.read_excel(both_xlsx_path, engine="openpyxl", sheet_name='NetDrift')
        both_net_drifts['Sex'] = 'Both'

        f_age_deviations = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='AgeDeviations')
        f_age_deviations['Sex'] = 'Female'
        f_period_deviations = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='PerDeviations')
        f_period_deviations['Sex'] = 'Female'
        f_cohort_deviations = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='CohDeviations')
        f_cohort_deviations['Sex'] = 'Female'
        f_longitudinal_age_curve = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='LongAge')
        f_longitudinal_age_curve['Sex'] = 'Female'
        f_cross_sectional_age_curve = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='CrossAge')
        f_cross_sectional_age_curve['Sex'] = 'Female'
        f_long_vs_cross_rr = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='Long2CrossRR')
        f_long_vs_cross_rr['Sex'] = 'Female'
        f_fitted_temporal_trends = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='FittedTemporalTrends')
        f_fitted_temporal_trends['Sex'] = 'Female'
        f_period_rr = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='PeriodRR')
        f_period_rr['Sex'] = 'Female'
        f_cohort_rr = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='CohortRR')
        f_cohort_rr['Sex'] = 'Female'
        f_local_drifts = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='LocalDrifts')
        f_local_drifts['Sex'] = 'Female'
        f_net_drifts = pd.read_excel(f_xlsx_path, engine="openpyxl", sheet_name='NetDrift')
        f_net_drifts['Sex'] = 'Female'

        m_age_deviations = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='AgeDeviations')
        m_age_deviations['Sex'] = 'Male'
        m_period_deviations = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='PerDeviations')
        m_period_deviations['Sex'] = 'Male'
        m_cohort_deviations = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='CohDeviations')
        m_cohort_deviations['Sex'] = 'Male'
        m_longitudinal_age_curve = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='LongAge')
        m_longitudinal_age_curve['Sex'] = 'Male'
        m_cross_sectional_age_curve = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='CrossAge')
        m_cross_sectional_age_curve['Sex'] = 'Male'
        m_long_vs_cross_rr = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='Long2CrossRR')
        m_long_vs_cross_rr['Sex'] = 'Male'
        m_fitted_temporal_trends = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='FittedTemporalTrends')
        m_fitted_temporal_trends['Sex'] = 'Male'
        m_period_rr = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='PeriodRR')
        m_period_rr['Sex'] = 'Male'
        m_cohort_rr = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='CohortRR')
        m_cohort_rr['Sex'] = 'Male'
        m_local_drifts = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='LocalDrifts')
        m_local_drifts['Sex'] = 'Male'
        m_net_drifts = pd.read_excel(m_xlsx_path, engine="openpyxl", sheet_name='NetDrift')
        m_net_drifts['Sex'] = 'Male'

        age_deviations = pd.concat([both_age_deviations, f_age_deviations, m_age_deviations], ignore_index=True)
        period_deviations = pd.concat(
            [both_period_deviations, f_period_deviations, m_period_deviations],
            ignore_index=True
        )
        cohort_deviations = pd.concat(
            [both_cohort_deviations, f_cohort_deviations, m_cohort_deviations],
            ignore_index=True
        )
        longitudinal_age_curve = pd.concat(
            [both_longitudinal_age_curve, f_longitudinal_age_curve, m_longitudinal_age_curve],
            ignore_index=True
        )
        cross_sectional_age_curve = pd.concat(
            [both_cross_sectional_age_curve, f_cross_sectional_age_curve, m_cross_sectional_age_curve],
            ignore_index=True
        )
        long_vs_cross_rr = pd.concat([both_long_vs_cross_rr, f_long_vs_cross_rr, m_long_vs_cross_rr], ignore_index=True)
        fitted_temporal_trends = pd.concat(
            [both_fitted_temporal_trends, f_fitted_temporal_trends, m_fitted_temporal_trends],
            ignore_index=True
        )
        period_rr = pd.concat([both_period_rr, f_period_rr, m_period_rr], ignore_index=True)
        cohort_rr = pd.concat([both_cohort_rr, f_cohort_rr, m_cohort_rr], ignore_index=True)
        local_drifts = pd.concat([both_local_drifts, f_local_drifts, m_local_drifts], ignore_index=True)
        net_drifts = pd.concat([both_net_drifts, f_net_drifts, m_net_drifts], ignore_index=True)

        plot_apc(
            age_deviations=age_deviations,
            period_deviations=period_deviations,
            cohort_deviations=cohort_deviations,
            longitudinal_age_curve=longitudinal_age_curve,
            cross_sectional_age_curve=cross_sectional_age_curve,
            long_vs_cross_rr=long_vs_cross_rr,
            fitted_temporal_trends=fitted_temporal_trends,
            period_rr=period_rr,
            cohort_rr=cohort_rr,
            local_drifts=local_drifts,
            net_drifts=net_drifts,
            output_dir=output_dir,
        )


if __name__ == '__main__':
    import os

    os.chdir(pathlib.Path(__file__).parent.parent)

    # df: pd.DataFrame = joblib.load(pathlib.Path("data") / "data_cause_of_death_or_injury.pkl")
    #
    # run(
    #     raw=df,
    #     title="Anxiety Disorders",
    #     description="Anxiety Disorders",
    #     name="AnxietyDisorders.APC.1990-2019",
    #     cause="Anxiety disorders",
    #     interval=5,
    #     years=(1990, 2019)
    # )
    run(
        both_xlsx_path=pathlib.Path("output/AnxietyDisorders.APC.1990-2019_1/APC.Analysis.Both.xlsx"),
        f_xlsx_path=pathlib.Path("output/AnxietyDisorders.APC.1990-2019_1/APC.Analysis.Female.xlsx"),
        m_xlsx_path=pathlib.Path("output/AnxietyDisorders.APC.1990-2019_1/APC.Analysis.Male.xlsx"),
    )
