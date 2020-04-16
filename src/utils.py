#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" utils.py

DESCRIPTION 

This file can also be imported as a module and contains the following functions:
    * 

TO DO:  
    *
"""

from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

from IPython.display import clear_output

def get_data(url: str, country: str) -> list:
    if country == 'spain':
        ccaa_codes = {'IB': 'Islas Baleares', 
                'CN': 'C.F. Navarra',
                'CT': 'Cataluña', 
                'VC': 'C. Valenciana', 
                'MD': 'C. de Madrid', 
                'AN': 'Andalucia', 
                'CL': 'Castilla y Leon', 
                'PV': 'Pais Vasco',
                'AS': 'Asturias', 
                'CB': 'Cantabria', 
                'CM': 'Castilla la Mancha', 
                'EX': 'Extremadura', 
                'NC': 'Canarias', 
                'RI': 'La Rioja', 
                'GA': 'Galicia',
                'AR': 'Aragon', 
                'MC': 'R. de Murcia', 
                'ML': 'Melilla', 
                'CE': 'Ceuta'}
        df = pd.read_csv(url, 
                         encoding = 'unicode_escape')
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True)
        df = df[df['CCAA'].str.len() <= 2]
        df['CCAA'] = df['CCAA'].map(ccaa_codes)
        df = df.fillna(0)
        df[df.columns[2:]] = df[df.columns[2:]].astype(int)
        df = df.rename(columns={list(df)[1]: 'date',
                        list(df)[0]: 'region',
                        list(df)[2]: 'cases'})
    elif country == 'usa':
        df = pd.read_csv(url,
                        usecols=[0,1,3],
                        parse_dates=['date'])
        df = df.rename(columns={list(df)[0]: 'date',
                list(df)[1]: 'region',
                list(df)[2]: 'cases'})
    df = df[df.columns[0:3]]
    return df, df.set_index(['region', 'date']).squeeze().sort_index(), df.groupby('region')['cases'].max()


def prepare_cases(df, rolling: str = 'median', rolling_window: int = 7) -> list:
    df_newcases = df.diff()
    if rolling == 'average':
        smoothed = df_newcases.rolling(rolling_window,
            win_type='gaussian',
            min_periods=1,
            center=True).mean(std=2).round()
    elif rolling == 'median':
        smoothed = df_newcases.rolling(rolling_window,
            min_periods=1,
            center=True).median().round()
    zeros = smoothed.index[smoothed.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed.index.get_loc(last_zero) + 1
    s_smoothed = smoothed.iloc[idx_start:]
    s_original = df_newcases.loc[smoothed.index]
    return s_original, s_smoothed 


def get_posteriors(s_smoothed: list, prior: str = 'gamma', window:int = 7, min_periods:int = 1, gamma: int = 0.25, r_t_max: int = 12) -> list:
    r_t_range = np.linspace(0, r_t_max, r_t_max*100+1)
    lam = s_smoothed[:-1].values * np.exp(gamma * (r_t_range[:, None] - 1))
    if prior == 'uniform':
        prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))
    elif prior == 'gamma':
        prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)

    likelihoods = pd.DataFrame(
        data = np.c_[prior0, sps.poisson.logpmf(s_smoothed[1:].values, lam)],
        index = r_t_range,
        columns = s_smoothed.index)
    posteriors = likelihoods.rolling(window,
                                     axis=1,
                                     min_periods=min_periods).sum()
    posteriors = np.exp(posteriors)
    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)
    return posteriors


def highest_density_interval(pmf, p=.95):
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col]) for col in pmf],
                            index=pmf.columns)
    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j<best[1]-best[0]):
                best = (i, i+j+1)
                break
    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=['Low', 'High'])


def plot_daily_cases(s_original, s_smoothed, ax, REGION):
    ax = s_original.plot(
        c='black',
        linestyle='-',
        linewidth=1.2,
        alpha=.8,
        label='Actual')

    ax = s_smoothed.plot(label='Trend',
                        c='gray',
                        linestyle='-',
                        linewidth=0.8,
                        alpha=.8,
#                        legend=True
                        )
    ax.get_figure().set_facecolor('w')
    ax.legend(fontsize=16)
    ax.set_title(f"{REGION} daily cases")
    return ax
    
def plot_posteriors(df_posteriors, ax, REGION):
    ax.plot(df_posteriors,
            lw=1,
            c='k',
            alpha=.3
           )
    ax.set_title(f'Posterior of $R_t$ in {REGION}')
    ax.set_xlabel('$R_t$')


def plot_rt(df_r0s, ax, t_cases, REGION):
    
    ax.set_title(f"$R_t$ of {REGION} (n={t_cases[REGION]})")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = df_r0s['ML'].index.get_level_values('date')
    values = df_r0s['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     df_r0s['Low'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      df_r0s['High'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25)
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0,3.5)
    ax.set_xlim(pd.Timestamp('2020-03-01'), df_r0s.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
    # ax.set_facecolor('w')