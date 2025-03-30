from __future__ import annotations

import pathlib
import typing

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as _colors

from gbd.common import (
    const as _const,
    utils as _utils,
)

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'

LOGGER = _utils.Logger()

LOCATIONS = [
    'Global',
    'High SDI',
    'High-middle SDI',
    'Middle SDI',
    'Low-middle SDI',
    'Low SDI',
    'Andean Latin America',
    'Australasia',
    'Caribbean',
    'Central Asia',
    'Central Europe',
    'Central Latin America',
    'East Asia',
    'Eastern Europe',
    'Eastern Sub-Saharan Africa',
    'High-income Asia Pacific',
    'High-income North America',
    'North Africa and Middle East',
    'Oceania',
    'South Asia',
    'Southeast Asia',
    'Southern Latin America',
    'Southern Sub-Saharan Africa',
    'Tropical Latin America',
    'Western Europe',
    'Western Sub-Saharan Africa'
]

SDI_REGIONS = [
    'High-middle SDI',
    'High SDI',
    'Middle SDI',
    'Low-middle SDI',
    'Low SDI',
    'Global'
]

SPEC_COLUMNS = ['age_name', 'sex_name', 'year', 'location_name']

COLORS_1 = ("#FF5512", "#FFE3D7")
COLORS_2 = ("lightblue", "lightyellow")

CMAP_1 = _colors.LinearSegmentedColormap.from_list("custom_cmap", COLORS_1, N=256)
CMAP_2 = _colors.LinearSegmentedColormap.from_list("custom_cmap", COLORS_2, N=256)


def plot_dalys(
    data: pd.DataFrame,
    *,
    spec_age: str = '15-19 years',
    year: int,
    output_dir: pathlib.Path,
):
    data = data[(data['val_rate'] > 0) | (data['val_num'] > 0)]
    data.loc[:, 'age_name'] = pd.Categorical(data['age_name'], _const.AGE_GROUPS, ordered=True)
    age_order = data['age_name'].unique()

    # Figure 1: Global Age-specific DALYs Rates by Sex -----------------------
    LOGGER.info(f"Plotting figures Global Age-specific DALYs Rates by Sex ({year})")
    global_data = data[data['location_name'] == 'Global']
    plt.figure(figsize=(12, 6))

    # male
    male_data = global_data[global_data['sex_name'] == 'Male']
    plt.plot(male_data['age_name'], male_data['val_rate'], marker='o', label='Male')
    plt.fill_between(
        male_data['age_name'],
        male_data['lower_rate'],
        male_data['upper_rate'],
        alpha=0.2,
        color='steelblue'
    )

    # female
    female_data = global_data[global_data['sex_name'] == 'Female']
    plt.plot(female_data['age_name'], female_data['val_rate'], marker='o', label='Female')
    plt.fill_between(
        female_data['age_name'],
        female_data['lower_rate'],
        female_data['upper_rate'],
        alpha=0.2,
        color='lightcoral'
    )

    plt.xlabel('Age Group')
    plt.ylabel('DALYs Rate (per 100,000)')
    plt.title(f'Global Age-specific DALYs Rates by Sex ({year})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1.global_dalys_age_gender.png')

    # Figure 2: Age-specific DALYs by SDI Regions and Sex --------------------
    LOGGER.info(f"Plotting figures Age-specific DALYs by SDI Regions and Sex ({year})")
    sdi_data = data.copy()
    f_sdi_data = sdi_data[sdi_data['sex_name'] == 'Female']
    m_sdi_data = sdi_data[sdi_data['sex_name'] == 'Male']

    f_pivot_table = f_sdi_data.pivot_table(index='age_name', columns='location_name', values='val_rate', observed=False)
    f_pivot_table = f_pivot_table.reindex(age_order)
    ax = f_pivot_table.plot(kind='bar', figsize=(14, 8), width=0.8, colormap='Set2')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('DALYs Rate (per 100,000)')
    ax.set_title(f'Age-specific DALYs Rates by SDI Regions - Female ({year})')
    plt.xticks(rotation=45)
    plt.legend(title='SDI Region', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2.female_dalys_age_sdi.png')

    m_pivot_table = m_sdi_data.pivot_table(index='age_name', columns='location_name', values='val_rate', observed=False)
    m_pivot_table = m_pivot_table.reindex(age_order)
    ax = m_pivot_table.plot(kind='bar', figsize=(14, 8), width=0.8, colormap='Set2')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('DALYs Rate (per 100,000)')
    ax.set_title(f'Age-specific DALYs Rates by SDI Regions - Female ({year})')
    plt.xticks(rotation=45)
    plt.legend(title='SDI Region', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2.male_dalys_age_sdi.png')

    # Figure 3: Age-specific Number of DALYs by SDI Regions and Global -----------
    LOGGER.info(f"Plotting figures Age-specific Number of DALYs by SDI Regions and Global ({year})")

    sdi_data_num = data[data['location_name'] != 'Global']
    f_sdi_data_num = sdi_data_num[sdi_data_num['sex_name'] == 'Female']
    m_sdi_data_num = sdi_data_num[sdi_data_num['sex_name'] == 'Male']

    f_pivot_table_num = f_sdi_data_num.pivot_table(
        index='age_name',
        columns='location_name',
        values='val_num',
        observed=False
    )
    f_pivot_table_num = f_pivot_table_num.reindex(age_order)

    ax = f_pivot_table_num.plot(
        kind='bar',
        stacked=True,
        figsize=(14, 8),
        width=0.8,
        colormap=CMAP_1,
        edgecolor='none'
    )
    ax.set_xlabel('Age Group')
    ax.set_ylabel('DALYs Number')
    ax.set_title(f'Age-specific Number of DALYs by SDI Regions and Global - Female ({year})')
    plt.xticks(rotation=45)
    plt.legend(title='SDI Region', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3.female_dalys_num_age_sdi.png')

    m_pivot_table_num = m_sdi_data_num.pivot_table(
        index='age_name',
        columns='location_name',
        values='val_num',
        observed=False
    )
    m_pivot_table_num = m_pivot_table_num.reindex(age_order)

    ax = m_pivot_table_num.plot(
        kind='bar',
        stacked=True,
        figsize=(14, 8),
        width=0.8,
        colormap=CMAP_1,
        edgecolor='none'
    )
    ax.set_xlabel('Age Group')
    ax.set_ylabel('DALYs Number')
    ax.set_title(f'Age-specific Number of DALYs by SDI Regions and Global - Male ({year})')
    plt.xticks(rotation=45)
    plt.legend(title='SDI Region', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3.male_dalys_num_age_sdi.png')

    # Figure 4: Difference in DALYs Rates by Age Group and SDI Region ------------
    LOGGER.info(f"Plotting figures Difference in DALYs Rates by Age Group and SDI Region ({year})")
    data_diff = data.copy()
    data_diff = data_diff[data_diff['location_name'] != 'Global']
    data_diff = data_diff.pivot_table(
        index='age_name', columns=['location_name', 'sex_name'], values='val_rate',
        observed=False
    )

    male_rates = data_diff.xs('Male', level='sex_name', axis=1).fillna(0)
    female_rates = data_diff.xs('Female', level='sex_name', axis=1).fillna(0)
    rate_diff = male_rates - female_rates

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        rate_diff.T,
        annot=True,
        fmt=".1f",
        cmap=CMAP_2,
        cbar_kws={'label': 'Rate Difference (Male - Female)'},
        annot_kws={"size": 10}
    )
    plt.title('Sex Difference in DALYs Rates by Age Group and SDI Region (2019)')
    plt.xlabel('Age Group')
    plt.ylabel('SDI Region')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4.gender_diff_dalys_age_sdi.png')

    # Figure 5: Specific Age Group DALYs Rate by SDI Region -----------------------
    LOGGER.info(f"Plotting figures Specific Age Group DALYs Rate by SDI Region ({year})")

    age_group_data = data[data['age_name'] == spec_age]
    age_group_data = age_group_data[age_group_data['location_name'] != 'Global']

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", 2)
    sns.barplot(
        x='location_name',
        y='val_rate',
        hue='sex_name',
        data=age_group_data,
        errorbar=None,
        palette=palette,
        alpha=0.7
    )
    plt.errorbar(
        x=np.arange(len(age_group_data['location_name'].unique())) - 0.2,
        y=age_group_data[age_group_data['sex_name'] == 'Male']['val_rate'],
        yerr=[age_group_data[age_group_data['sex_name'] == 'Male']['val_rate'] -
              age_group_data[age_group_data['sex_name'] == 'Male']['lower_rate'],
              age_group_data[age_group_data['sex_name'] == 'Male']['upper_rate'] -
              age_group_data[age_group_data['sex_name'] == 'Male']['val_rate']],
        fmt='none',
        c='brown',
        capsize=5
    )

    plt.errorbar(
        x=np.arange(len(age_group_data['location_name'].unique())) + 0.2,
        y=age_group_data[age_group_data['sex_name'] == 'Female']['val_rate'],
        yerr=[age_group_data[age_group_data['sex_name'] == 'Female']['val_rate'] -
              age_group_data[age_group_data['sex_name'] == 'Female']['lower_rate'],
              age_group_data[age_group_data['sex_name'] == 'Female']['upper_rate'] -
              age_group_data[age_group_data['sex_name'] == 'Female']['val_rate']],
        fmt='none',
        c='brown',
        capsize=5
    )

    plt.xlabel('SDI Region')
    plt.ylabel('DALYs Rate (per 100,000)')
    plt.title(f'DALYs Rates in {spec_age} by SDI Region and Sex (2019)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / f'fig5.{spec_age}_dalys_rate_sdi.png')


def plot_asdr_aapc(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    file_pattern: str = '*.Export.AAPC.txt'
):
    data_list = []
    for file in data_dir.glob(file_pattern):
        LOGGER.info(f"Processing {file.name}")
        data_ = pd.read_csv(file)
        data_['Location'] = file.name.split('.')[0]
        data_list.append(data_)

    data = pd.concat(data_list, ignore_index=True)[['Location', 'AAPC', 'AAPC C.I. Low', 'AAPC C.I. High']]
    data.loc[:, 'Location'] = pd.Categorical(data['Location'], LOCATIONS, ordered=True)
    data = data.sort_values('Location').reset_index(drop=True)
    data['Error Low'] = data['AAPC'] - data['AAPC C.I. Low']
    data['Error High'] = data['AAPC C.I. High'] - data['AAPC']

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Location',
        y='AAPC',
        data=data,
        color='skyblue',
    )
    plt.errorbar(
        x=np.arange(len(data['Location'])),
        y=data['AAPC'],
        yerr=[data['Error Low'], data['Error High']],
        fmt='none',
        c='brown',
        capsize=5
    )

    plt.xlabel('Location')
    plt.ylabel('AAPC')
    plt.title('ASDR Average Annual Percent Change (AAPC) by Location')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6.aapc_location.png')


def run(
    raw: pd.DataFrame,
    *,
    asdr_root: typing.Optional[pathlib.Path] = None,
    cause: str = "Anxiety disorders",
    rei: str = "Bullying victimization",
    year: int = 2019,
    output_dir: pathlib.Path = pathlib.Path("output"),
    name: str = "exp",
):
    # Initialize -------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _utils.unique_path(output_dir, name)

    # Process Data -----------------------------------------------------------
    dalys_number_df = raw[(raw['metric_name'] == 'Number') &  # number of cases
                          (raw['year'] == year) &  # year
                          (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                          (raw['cause_name'] == cause) &  # cause
                          (raw['rei_name'] == rei) &  # risk factor
                          (raw['sex_name'] != 'Both') &  # exclude 'Both'
                          (raw['location_name'].isin(SDI_REGIONS)) &  # SDI regions
                          (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')]  # DALYs

    dalys_rate_df = raw[(raw['metric_name'] == 'Rate') &  # rate of cases
                        (raw['year'] == year) &  # year
                        (raw['age_name'].isin(_const.AGE_GROUPS)) &  # age groups
                        (raw['cause_name'] == cause) &  # cause
                        (raw['rei_name'] == rei) &  # risk factor
                        (raw['sex_name'] != 'Both') &  # exclude 'Both'
                        (raw['location_name'].isin(SDI_REGIONS)) &  # SDI regions
                        (raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')]  # DALYs

    merged_df = pd.merge(
        dalys_number_df, dalys_rate_df, on=SPEC_COLUMNS, suffixes=('_num', '_rate')
    )[SPEC_COLUMNS + ['val_num', 'val_rate', 'upper_num', 'upper_rate', 'lower_num', 'lower_rate']]
    result_df = merged_df.copy()
    result_df.columns = ['Age Group', 'Sex', 'Year', 'Location', 'Number', 'Rate', "Number Upper", "Rate Upper",
                         "Number Lower", "Rate Lower"]
    result_df.to_csv(
        output_dir / 'result.csv',
        index=False,
    )
    plot_dalys(merged_df, year=year, output_dir=output_dir)

    if asdr_root:
        plot_asdr_aapc(asdr_root, output_dir)
    LOGGER.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    import os

    os.chdir(pathlib.Path(__file__).parent.parent)

    df: pd.DataFrame = joblib.load(pathlib.Path("data") / "data_risk_factors.pkl")

    run(
        df,
        name="AnxietyDisorders.DALYs&AAPC.2021",
        asdr_root=pathlib.Path("output/AnxietyDisorders.Statistics&YearTrend&LocationASDR.1992-2021/Location ASDR"),
        year=2021,
    )
