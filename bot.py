from argparse import ArgumentParser
from asyncio import run
from asyncpg import connect
from collections import OrderedDict
from datetime import datetime, timedelta
from json import dump, load
from math import pi, sin, cos
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mtick
from platform import system
from requests import get
from tweepy import OAuthHandler, API
import traceback


# Multi-day, use gte
battles_query = '''
    SELECT sum(battles), console, _date
    FROM diff_battles
    WHERE _date >= '{}' AND _date < '{}'
    GROUP BY console, _date
    ORDER BY console, _date ASC
'''

# Multi-day, use gte
players_query = '''
    SELECT count(account_id), console, _date
    FROM diff_battles
    WHERE _date >= '{}' AND _date < '{}'
    GROUP BY console, _date
    ORDER BY console, _date ASC
'''

unique_count_query = '''
    SELECT count(distinct(account_id)), console
    FROM diff_battles
    WHERE _date >= '{}' AND _date < '{}'
    GROUP BY console
    ORDER BY console
'''

new_players_query = '''
    SELECT count(account_id), console, created_at::date
    FROM players
    WHERE created_at >= '{}' AND created_at < '{}'
    GROUP BY console, created_at::date
    ORDER BY console, created_at::date
'''

personal_players_query = '''
    SELECT diff_battles.account_id, players.created_at, players.console
    FROM diff_battles, players
    WHERE diff_battles._date >= '{}' AND diff_battles._date < '{}'
        AND diff_battles.account_id = players.account_id
'''

accounts_per_battles_range_query = '''
    SELECT "Battle Buckets" AS "Range", count("Battle Buckets") AS "Count of Players", console
    FROM (
        SELECT console,
        CASE
            WHEN diff_battles.battles >= 1  AND diff_battles.battles < 5  THEN '1-4'
            WHEN diff_battles.battles >= 5  AND diff_battles.battles < 10 THEN '5-9'
            WHEN diff_battles.battles >= 10 AND diff_battles.battles < 20 THEN '10-19'
            WHEN diff_battles.battles >= 20 AND diff_battles.battles < 30 THEN '20-29'
            WHEN diff_battles.battles >= 30 AND diff_battles.battles < 40 THEN '30-39'
            WHEN diff_battles.battles >= 40 AND diff_battles.battles < 50 THEN '40-49'
            ELSE '50-*'
            END AS "Battle Buckets"
        FROM diff_battles
        WHERE _date >= '{}' AND _date < '{}'
    ) p
    GROUP BY console, "Battle Buckets"
'''

five_battles_a_day_query = '''
    SELECT count(*), console, _date
    FROM diff_battles
    WHERE battles >= 5 AND _date >= '{}' AND _date < '{}'
    GROUP BY console, _date
    ORDER BY _date, console
'''

popular_tanks_query = '''
    SELECT short_name, era, t_b.tank_id, console, tot_batt
    FROM (
        SELECT tank_id, console, sum(battles) AS tot_batt, ROW_NUMBER() OVER (PARTITION BY console ORDER BY sum(battles) DESC) AS r_id
        FROM diff_tanks
        WHERE _date >= '{}' AND _date < '{}'
        GROUP BY console, tank_id
        ORDER BY console, tot_batt DESC
    ) AS t_b, tanks as t
    WHERE t.tank_id = t_b.tank_id
'''

mode_battles_query = '''
    SELECT sum(battles), console, _date,
    CASE
        WHEN tanks.era <> '' THEN 'CW'
        ELSE 'WW2'
        END AS "t_era"
    FROM diff_tanks, tanks
    WHERE _date >= '{}' AND _date < '{}' AND tanks.tank_id = diff_tanks.tank_id
    GROUP BY console, t_era, _date
'''

BATTLES_PNG = '{}/tmp/battles.png'.format('.' if system() == 'Windows' else '')
PLAYERS_PNG = '{}/tmp/players.png'.format('.' if system() == 'Windows' else '')
NEWPLAYERS_PNG = '{}/tmp/newplayers.png'.format('.' if system() == 'Windows' else '')
AVERAGE_PNG = '{}/tmp/average.png'.format('.' if system() == 'Windows' else '')
ACCOUNTAGE_PNG = '{}/tmp/accountage.png'.format('.' if system() == 'Windows' else '')
BATTLERANGE_PNG = '{}/tmp/battlerange.png'.format('.' if system() == 'Windows' else '')
FIVEADAY_PNG = '{}/tmp/fiveaday.png'.format('.' if system() == 'Windows' else '')
PLAYERSLONG_PNG = '{}/tmp/playerslong.png'.format('.' if system() == 'Windows' else '')
BATTLESLONG_PNG = '{}/tmp/battleslong.png'.format('.' if system() == 'Windows' else '')
AVERAGELONG_PNG = '{}/tmp/averagelong.png'.format('.' if system() == 'Windows' else '')
MODEBREAKDOWN_PNG = '{}/tmp/modebreakdown.png'.format('.' if system() == 'Windows' else '')
MODEBREAKDOWNLONG_PNG = '{}/tmp/modebreakdownlong.png'.format('.' if system() == 'Windows' else '')
MODEBREAKDOWNPERCENT_PNG = '{}/tmp/modebreakdownpercent.png'.format('.' if system() == 'Windows' else '')
MODEBREAKDOWNPERCENTLONG_PNG = '{}/tmp/modebreakdownpercentlong.png'.format('.' if system() == 'Windows' else '')

def manage_config(mode, filename='config.json'):
    if mode == 'read':
        with open(filename) as f:
            return load(f)
    elif mode == 'create':
        with open(filename, 'w') as f:
            dump(
                {
                    'days': 14,
                    'long term': 90,
                    'omit errors long term': True,
                    'twitter': {
                        'api key': '',
                        'api secret key': '',
                        'access token': '',
                        'access token secret': '',
                        'message': "Today's update on the active player count and total battles per platform for #worldoftanksconsole."
                    },
                    'database': {
                        'host': 'localhost',
                        'user': 'username',
                        'password': 'password',
                        'database': 'battletracker',
                        'port': 5432
                    },
                    'unique': [7, 14, 30],
                    'account age': [7, 30, 90, 180, 365, 730, 1095, 1460, 1825],
                    'watermark text': '@WOTC_Tracker',
                    'wg api key': 'DEMO'
                }
            )


async def query_for_graphs(config, conn):
    now = datetime.utcnow()
    now_s = now.strftime('%Y-%m-%d')
    then = now - timedelta(days=config['days'])
    then_s = then.strftime('%Y-%m-%d')

    # Filter numbers
    dates = [(then + timedelta(days=i)).date() for i in range((now - then).days)]
    battles = {
        'xbox': OrderedDict((k, 0) for k in dates),
        'ps': OrderedDict((k, 0) for k in dates)
    }
    players = {
        'xbox': OrderedDict((k, 0) for k in dates),
        'ps': OrderedDict((k, 0) for k in dates)
    }
    newplayers = {
        'xbox': OrderedDict((k, 0) for k in dates),
        'ps': OrderedDict((k, 0) for k in dates)
    }
    averages = {
        'xbox': OrderedDict((k, 0) for k in dates),
        'ps': OrderedDict((k, 0) for k in dates)
    }

    # Query database
    results = await conn.fetch(battles_query.format(then_s, now_s))
    for record in results:
        battles[record['console']][record['_date']] = record['sum']

    results = await conn.fetch(players_query.format(then_s, now_s))
    for record in results:
        players[record['console']][record['_date']] = record['count']

    results = await conn.fetch(new_players_query.format(then_s, now_s))
    for record in results:
        newplayers[record['console']][record['created_at']] = record['count']

    for d, b, p in zip(dates, battles['xbox'].values(), players['xbox'].values()):
        averages['xbox'][d] = (b / p) if p != 0 else 0

    for d, b, p in zip(dates, battles['ps'].values(), players['ps'].values()):
        averages['ps'][d] = (b / p) if p != 0 else 0

    return (
        dates, battles['xbox'], battles['ps'], 
        players['xbox'], players['ps'], newplayers['xbox'], newplayers['ps'],
        averages['xbox'], averages['ps']
    )


async def query_for_unique(config, conn):
    now = datetime.utcnow()
    unique = {
        'Xbox': OrderedDict((d, 0) for d in config['unique']), 
        'Playstation': OrderedDict((d, 0) for d in config['unique'])
    }
    for earliest in config['unique']:
        results = await conn.fetch(unique_count_query.format((now - timedelta(days=earliest)).strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))
        for bucket in results:
            unique['Xbox' if bucket['console'] == 'xbox' else 'Playstation'][earliest] = bucket['count']
    return unique


def create_activity_graphs(dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_xbox, newplayers_ps, averages_xbox, averages_ps, watermark_text='@WOTC_Tracker'):
    # Players PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('Active Accounts Per Platform')
    # ax1 = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates, ha='right')
    ax1.plot(dates, players_xbox.values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, players_ps.values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.savefig(PLAYERS_PNG)
    del fig
    # Battles PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('Total Battles Per Platform')
    # ax = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates, ha='right')
    ax1.plot(dates, battles_xbox.values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, battles_ps.values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.savefig(BATTLES_PNG)
    del fig
    # New Players PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('New Accounts Per Platform')
    # ax = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates, ha='right')
    ax1.plot(dates, newplayers_xbox.values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, newplayers_ps.values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.savefig(NEWPLAYERS_PNG)
    del fig
    # Averages PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('Average Battles Played Per Account Per Platform')
    # ax = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates, ha='right')
    ax1.plot(dates, averages_xbox.values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, averages_ps.values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.savefig(AVERAGE_PNG)
    del fig


async def query_for_active_accounts(config, conn):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    
    player_info = await conn.fetch(personal_players_query.format(then.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))

    buckets = {
        "xbox": OrderedDict((v, 0) for v in sorted(config['account age'])),
        "ps": OrderedDict((v, 0) for v in sorted(config['account age'])),
        "all": OrderedDict((v, 0) for v in sorted(config['account age']))
    }

    # Sum account ages based on range of age
    buckets['xbox']['other'] = 0
    buckets['ps']['other'] = 0
    buckets['all']['other'] = 0
    for player in player_info:
        delta = now - player['created_at']
        for key in buckets['all'].keys():
            if not isinstance(key, int):
                buckets['all'][key] += 1
                buckets[player['console']][key] += 1
                break
            elif delta.total_seconds() <= (key * 24 * 60 * 60):
                buckets['all'][key] += 1
                buckets[player['console']][key] += 1
                break
    return buckets


def calc_label(value):
    if value < 7:
        return '{} day{}'.format(value, '' if value == 1 else 's')
    elif 7 <= value < 30:
        return '{} week{}'.format(value // 7, '' if value // 7 == 1 else 's')
    elif 30 <= value < 365:
        return '{} month{}'.format(value // 30, '' if value // 30 == 1 else 's')
    else:
        return '{} year{}'.format(value // 365, '' if value // 365 == 1 else 's')


def calc_angle(wedge):
    return (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1


def create_account_age_chart(buckets, watermark_text='@WOTC_Tracker'):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    then = datetime.utcnow() - timedelta(days=1)
    fig.suptitle("Breakdown of active accounts by account age for {}".format(then.strftime('%Y-%m-%d')))
    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=10)
    ax1.axis('equal')
    size = 0.125

    outer_labels = []
    prev = 0
    for key in buckets['all'].keys():
        if not isinstance(key, int):
            outer_labels.append('>' + calc_label(prev))
        else:
            outer_labels.append('{} - {}'.format(calc_label(prev), calc_label(key)))
            prev = key

    # Outer pie chart
    outer_cmap = plt.get_cmap("binary")
    outer_colors = outer_cmap([i * 10 for i in range(10, len(buckets['all'].keys()) + 11)])
    outer_wedges, outer_text, outer_autotext = ax1.pie(
        buckets['all'].values(),
        explode=[0.1 for __ in outer_labels],
        radius=1,
        colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='%1.1f%%',
        pctdistance=1.1
        #labels=outer_labels
    )

    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle='-'), bbox=bbox_props, zorder=0, va='center')
    for i, wedge in enumerate(outer_wedges):
        angle = calc_angle(wedge)
        y = sin(angle * (pi / 180))
        x = cos(angle * (pi / 180))
        align = 'right' if x < 0 else 'left'
        connectionstyle = 'angle,angleA=0,angleB={}'.format(angle)
        kw['arrowprops'].update({'connectionstyle': connectionstyle})
        ax1.annotate(
            outer_labels[i],
            xy=(x, y),
            xytext=(1.35*(-1 if x < 0 else 1), 1.4*y),
            horizontalalignment=align,
            **kw
        )

    # Inner pie chart
    inner_cmap = plt.get_cmap("tab20c")
    pie_flat = list(zip(buckets['xbox'].values(), buckets['ps'].values()))
    inner_labels = []
    for pair in pie_flat:
        inner_labels.extend(['xbox', 'ps'])
    inner_colors = inner_cmap([1 if console == 'ps' else 9 for console in inner_labels])
    inner_wedges, inner_text, inner_autotext = ax1.pie(
        [item for sublist in pie_flat for item in sublist],
        explode=[0.1 for __ in inner_labels],
        radius=1.05-size,
        colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='',
        pctdistance=0.9
    )

    # Replace inner text with actual values
    for i, label, wedge, text in zip(range(len(inner_wedges)), inner_labels, inner_wedges, inner_autotext):
        text.set_text(buckets[label]['other' if i // 2 > len(buckets['all'].keys()) - 1 else list(buckets['all'].keys())[i // 2]])
        angle = calc_angle(wedge)
        if 90 < angle < 270:
            angle += 180
        text.set_rotation(angle)

    # Patch inner wedges to group together in explosion
    # Influenced by: https://stackoverflow.com/a/20556088/1993468
    groups = [[i, i+1] for i in range(0, len(inner_wedges), 2)]
    radfraction = 0.1
    for group in groups:
        angle = ((inner_wedges[group[-1]].theta2 + inner_wedges[group[0]].theta1)/2) * (pi / 180)
        for g in group:
            wedge = inner_wedges[g]
            wedge.set_center((radfraction * wedge.r * cos(angle), radfraction * wedge.r * sin(angle)))

    # Add subplot in second row, below nested pie chart
    ax2 = plt.subplot2grid((11, 1), (10, 0))
    ax2.axhline(color='black', y=0)
    # Xbox, Playstation
    totals = [sum(buckets['xbox'].values()), sum(buckets['ps'].values()), sum(buckets['all'].values())]
    ypos = -0.18
    bottom = 0
    height = 0.1
    for i in range(len(totals) - 1):
        width = totals[i] / totals[-1]
        ax2.barh(ypos, width, height, left=bottom, color=inner_colors[i])
        xpos = bottom + ax2.patches[i].get_width() / 2
        bottom += width
        ax2.text(xpos, ypos, '{} ({:.1f}%)'.format(totals[i], (totals[i] / totals[-1]) * 100), ha='center', va='center')

    ax2.axis('off')
    ax2.set_title('Total Active Players', y=0.325)
    ax2.set_xlim(0, 1)

    ax1.legend(inner_wedges[-2:], ['xbox', 'ps'], loc='lower right')
    fig.text(0.5, 0.5, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(ACCOUNTAGE_PNG)
    del fig


async def query_for_accounts_by_battles(config, conn):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    
    response = await conn.fetch(accounts_per_battles_range_query.format(then.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))
    keys = sorted(set([r['Range'] for r in response]), key=lambda k: int(k.split('-')[0]))

    buckets = {
        "xbox": OrderedDict((k, 0) for k in keys),
        "ps": OrderedDict((k, 0) for k in keys),
        "all": OrderedDict((k, 0) for k in keys),
    }
    for record in response:
        buckets['all'][record['Range']] += record['Count of Players']
        buckets[record['console']][record['Range']] = record['Count of Players']
    return buckets


def create_accounts_by_battles_chart(buckets, watermark_text='@WOTC_Tracker'):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    then = datetime.utcnow() - timedelta(days=1)
    fig.suptitle("Breakdown of accounts by number of battles played for {}".format(then.strftime('%Y-%m-%d')))
    # ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=10)
    ax1 = plt.axes()
    ax1.axis('equal')
    size = 0.125

    outer_labels = [b + ' battles' for b in buckets['all'].keys()]

    # Outer pie chart
    outer_cmap = plt.get_cmap("binary")
    outer_colors = outer_cmap([i * 10 for i in range(10, len(buckets['all'].keys()) + 11)])
    outer_wedges, outer_text, outer_autotext = ax1.pie(
        buckets['all'].values(),
        explode=[0.1 for __ in outer_labels],
        radius=1,
        colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='%1.1f%%',
        pctdistance=1.1
        #labels=outer_labels
    )

    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle='-'), bbox=bbox_props, zorder=0, va='center')
    for i, wedge in enumerate(outer_wedges):
        angle = calc_angle(wedge)
        y = sin(angle * (pi / 180))
        x = cos(angle * (pi / 180))
        align = 'right' if x < 0 else 'left'
        connectionstyle = 'angle,angleA=0,angleB={}'.format(angle)
        kw['arrowprops'].update({'connectionstyle': connectionstyle})
        ax1.annotate(
            outer_labels[i],
            xy=(x, y),
            xytext=(1.35*(-1 if x < 0 else 1), 1.4*y),
            horizontalalignment=align,
            **kw
        )

    # Inner pie chart
    inner_cmap = plt.get_cmap("tab20c")
    pie_flat = list(zip(buckets['xbox'].values(), buckets['ps'].values()))
    inner_labels = []
    for pair in pie_flat:
        inner_labels.extend(['xbox', 'ps'])
    inner_colors = inner_cmap([1 if console == 'ps' else 9 for console in inner_labels])
    inner_wedges, inner_text, inner_autotext = ax1.pie(
        [item for sublist in pie_flat for item in sublist],
        explode=[0.1 for __ in inner_labels],
        radius=1.05-size,
        colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='',
        pctdistance=0.9
    )

    # Replace inner text with actual values
    for i, label, wedge, text in zip(range(len(inner_wedges)), inner_labels, inner_wedges, inner_autotext):
        text.set_text(buckets[label]['other' if i // 2 > len(buckets['all'].keys()) - 1 else list(buckets['all'].keys())[i // 2]])
        angle = calc_angle(wedge)
        if 90 < angle < 270:
            angle += 180
        text.set_rotation(angle)

    # Patch inner wedges to group together in explosion
    # Influenced by: https://stackoverflow.com/a/20556088/1993468
    groups = [[i, i+1] for i in range(0, len(inner_wedges), 2)]
    radfraction = 0.1
    for group in groups:
        angle = ((inner_wedges[group[-1]].theta2 + inner_wedges[group[0]].theta1)/2) * (pi / 180)
        for g in group:
            wedge = inner_wedges[g]
            wedge.set_center((radfraction * wedge.r * cos(angle), radfraction * wedge.r * sin(angle)))

    ax1.legend(inner_wedges[-2:], ['xbox', 'ps'], loc='lower right')
    fig.text(0.5, 0.5, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(BATTLERANGE_PNG)
    del fig


async def query_five_battles_a_day_minimum(config, conn):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'])

    response = await conn.fetch(five_battles_a_day_query.format(then.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))
    keys = [(then + timedelta(days=i)).date() for i in range((now - then).days)]

    buckets = {
        "xbox": OrderedDict((k, 0) for k in keys),
        "ps": OrderedDict((k, 0) for k in keys),
        "all": OrderedDict((k, 0) for k in keys)
    }

    for record in response:
        buckets['all'][record['_date']] += record['count']
        buckets[record['console']][record['_date']] += record['count']

    return buckets


# Requested by Khorne Dog in the forums
def create_five_battles_minimum_chart(buckets, watermark_text='@WOTC_Tracker'):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle("Number of accounts having played at least 5 battles")
    ax1 = fig.add_subplot(111)

    width = 0.25
    keys = [datetime(d.year, d.month, d.day) for d in buckets['all'].keys()]  # datetime.date to datetime.datetime
    xkeys = [d - timedelta(hours=3) for d in keys]
    pkeys = [d + timedelta(hours=3) for d in keys]
    xbox_bars = ax1.bar(xkeys, buckets['xbox'].values(), width=width, color='g')
    ps_bars = ax1.bar(pkeys, buckets['ps'].values(), width=width, color='b')
    ax1.table(
        cellText=[
            list(buckets['xbox'].values()),
            list(buckets['ps'].values()),
            list(buckets['all'].values())],
        rowLabels=['xbox', 'ps', 'all'],
        colLabels=[d.strftime('%Y-%m-%d') for d in keys],
        loc='bottom')
    ax1.set_ylabel('Accounts')
    ax1.set_xticks([])
    ax1.legend((xbox_bars[0], ps_bars[0]), ('xbox', 'ps'))
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(FIVEADAY_PNG)


async def query_long_term_data(config, conn, filter_server_failures=True):
    now = datetime.utcnow()
    then = now - timedelta(days=config.get('long term', 90) + 1)
    dates = [(then + timedelta(days=i)).date() for i in range((now - then).days)]

    players = await conn.fetch(players_query.format(then.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))
    battles = await conn.fetch(battles_query.format(then.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))

    players_buckets = {
        "xbox": OrderedDict((k, 0) for k in dates),
        "ps": OrderedDict((k, 0) for k in dates),
        "all": OrderedDict((k, 0) for k in dates)
    }

    battles_buckets = {
        "xbox": OrderedDict((k, 0) for k in dates),
        "ps": OrderedDict((k, 0) for k in dates),
        "all": OrderedDict((k, 0) for k in dates)
    }

    average_battles_per_day_buckets = {
        "xbox": OrderedDict((k, 0) for k in dates),
        "ps": OrderedDict((k, 0) for k in dates),
        "all": OrderedDict((k, 0) for k in dates)
    }

    for record in battles:
        battles_buckets['all'][record['_date']] += record['sum']
        battles_buckets[record['console']][record['_date']] = record['sum']

    for record in players:
        players_buckets['all'][record['_date']] += record['count']
        players_buckets[record['console']][record['_date']] = record['count']

    if filter_server_failures:
        skip_next = False
        for key, value in players_buckets['ps'].items():
            # 20,000 is way below normal. Sometimes the server dies partway through. This day should be skipped
            if value < 20000:
                players_buckets['xbox'][key] = None
                players_buckets['ps'][key] = None
                players_buckets['all'][key] = None
                battles_buckets['xbox'][key] = None
                battles_buckets['ps'][key] = None
                battles_buckets['all'][key] = None
                skip_next = True
            elif skip_next:
                players_buckets['xbox'][key] = None
                players_buckets['ps'][key] = None
                players_buckets['all'][key] = None
                battles_buckets['xbox'][key] = None
                battles_buckets['ps'][key] = None
                battles_buckets['all'][key] = None
                skip_next = False

    for key in players_buckets['all'].keys():
        if players_buckets['xbox'][key] is None:
            average_battles_per_day_buckets['all'][key] = None
            average_battles_per_day_buckets['xbox'][key] = None
            average_battles_per_day_buckets['ps'][key] = None
        else:
            average_battles_per_day_buckets['xbox'][key] = battles_buckets['xbox'][key] / players_buckets['xbox'][key]
            average_battles_per_day_buckets['ps'][key] = battles_buckets['ps'][key] / players_buckets['ps'][key]
            average_battles_per_day_buckets['all'][key] = (battles_buckets['xbox'][key] + battles_buckets['ps'][key]) / (players_buckets['xbox'][key] + players_buckets['ps'][key])

    return players_buckets, battles_buckets, average_battles_per_day_buckets


def create_long_term_charts(players_buckets, battles_buckets, average_battles_per_day_buckets, watermark_text='@WOTC_Tracker'):
    dates = list(players_buckets['all'].keys())
    # Players PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Active Accounts Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates)
    ax1.plot(dates, players_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, players_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, -0.15, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    fig.autofmt_xdate()
    # ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(PLAYERSLONG_PNG)
    del fig
    # Battles PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Total Battles Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates)
    ax1.plot(dates, battles_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, battles_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, -0.15, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    fig.autofmt_xdate()
    # ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(BATTLESLONG_PNG)
    del fig
    # Average PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Average Battles Played Per Account Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates)
    ax1.plot(dates, average_battles_per_day_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, average_battles_per_day_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, -0.15, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    fig.autofmt_xdate()
    # ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(AVERAGELONG_PNG)
    del fig


def upload_long_term_charts(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    playerslong = api.media_upload(PLAYERSLONG_PNG)
    battleslong = api.media_upload(BATTLESLONG_PNG)
    averagelong = api.media_upload(AVERAGELONG_PNG)
    api.update_status(
        status='Long-term view of active accounts, with downtime and multi-day catchup errors omitted',
        media_ids=[playerslong.media_id, battleslong.media_id, averagelong.media_id]
    )


def upload_long_term_mode_charts(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    modelong = api.media_upload(MODEBREAKDOWNLONG_PNG)
    percentlong = api.media_upload(MODEBREAKDOWNPERCENTLONG_PNG)
    api.update_status(
        status='Long-term view of battles per mode',
        media_ids=[modelong.media_id, percentlong.media_id]
    )


def upload_activity_graphs_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    battles = api.media_upload(BATTLES_PNG)
    players = api.media_upload(PLAYERS_PNG)
    newplayers = api.media_upload(NEWPLAYERS_PNG)
    averages = api.media_upload(AVERAGE_PNG)
    api.update_status(
        status=config['twitter']['message'],
        media_ids=[players.media_id, battles.media_id, newplayers.media_id, averages.media_id]
    )


def upload_account_age_graph_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    accountage = api.media_upload(ACCOUNTAGE_PNG)
    api.update_status(
        status='Breakdown of active accounts by age per platform on #worldoftanksconsole',
        media_ids=[accountage.media_id]
    )


def upload_accounts_by_battles_chart_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    battlerange = api.media_upload(BATTLERANGE_PNG)
    api.update_status(
        status='Breakdown of accounts by number of battles played on #worldoftanksconsole',
        media_ids=[battlerange.media_id]
    )


def upload_five_battles_minimum_chart_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    fiveaday = api.media_upload(FIVEADAY_PNG)
    api.update_status(
        status='Filtering accounts per day with 5 battles minimum on #worldoftanksconsole',
        media_ids=[fiveaday.media_id]
    )


def share_unique_with_twitter(config, unique):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    status = 'Unique Active Accounts For {} Over Time\n{}'
    formatting = '{} days: {}'
    for platform in unique:
        api.update_status(
            status=status.format(
                platform,
                '\n'.join(map(lambda l: formatting.format(*l), unique[platform].items()))
            )
        )


async def query_for_top_tanks(config, conn):
    now = datetime.utcnow()
    then = now - timedelta(days=1)

    response = await conn.fetch(popular_tanks_query.format(then.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))
    buckets = {
        'CW': {
            'Xbox': OrderedDict(),
            'Playstation': OrderedDict()
        },
        'WW2': {
            'Xbox': OrderedDict(),
            'Playstation': OrderedDict()
        }
    }
    for record in response:
        console = 'Xbox' if record['console'] == 'xbox' else 'Playstation'
        era = 'CW' if record['era'] != '' else 'WW2'
        if len(buckets[era][console]) < 5:
            buckets[era][console][record['short_name']] = record['tot_batt']
    return buckets


def share_top_tanks(config, era, top, day):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    for platform, tanks in top.items():
        status = "Most used {} tanks on {} for {}\n{}"
        formatting = '{}: {} battles'
        api.update_status(
            status=status.format(
                era,
                platform.capitalize(),
                day,
                '\n'.join([formatting.format(tank, battles) for tank, battles in tanks.items()])
            )
        )


async def query_for_mode_battles_difference(config, conn, long_term=False):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'] if not long_term else config.get('long term', 90))

    battles = await conn.fetch(mode_battles_query.format(then.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))
    dates = [(then + timedelta(days=i)).date() for i in range((now - then).days)]
    # Filter numbers
    buckets = {
        'WW2': {
            'xbox': OrderedDict((d, 0) for d in dates),
            'ps': OrderedDict((d, 0) for d in dates)
        },
        'CW': {
            'xbox': OrderedDict((d, 0) for d in dates),
            'ps': OrderedDict((d, 0) for d in dates)
        },
        'percent': {
            'xbox': OrderedDict((d, 0) for d in dates),
            'ps': OrderedDict((d, 0) for d in dates)
        }
    }

    for record in battles:
        buckets[record['t_era']][record['console']][record['_date']] = record['sum']

    for key in buckets['percent']:
        for d in dates:
            total = buckets['CW'][key][d] + buckets['WW2'][key][d]
            if total != 0:
                buckets['percent'][key][d] = buckets['CW'][key][d] / total

    return dates, buckets


def create_mode_difference_graph(dates, buckets, long_term=False, watermark_text='@WOTC_Tracker'):
    # Mode PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150) if not long_term else plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Estimated breakdown of battles between CW and WW2, per platform{}'.format('' if not long_term else ' (long term'))
    # ax1 = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticks(dates, ha='right')
    ax1.plot(dates, buckets['WW2']['xbox'].values(), color='darkgreen', linewidth=2, label='WW2: Xbox')
    ax1.plot(dates, buckets['CW']['xbox'].values(), color='lightgreen', linewidth=2, label='CW: Xbox')
    ax1.plot(dates, buckets['WW2']['ps'].values(), color='darkblue', linewidth=2, label='WW2: Playstation')
    ax1.plot(dates, buckets['CW']['ps'].values(), color='lightblue', linewidth=2, label='CW: Playstation')
    ax1.set_ylim(bottom=0)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.savefig(MODEBREAKDOWN_PNG if not long_term else MODEBREAKDOWNLONG_PNG)
    del fig
    # Mode Percent PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150) if not long_term else plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Estimated percentage of battles taking place in CW, per platform{}'.format('' if not long_term else ' (long term)'))
    # ax1 = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_xticks(dates, ha='right')
    ax1.plot(dates, buckets['percent']['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, buckets['percent']['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.savefig(MODEBREAKDOWNPERCENT_PNG if not long_term else MODEBREAKDOWNPERCENTLONG_PNG)
    del fig


def upload_mode_breakdown_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    battles = api.media_upload(MODEBREAKDOWN_PNG)
    percent = api.media_upload(MODEBREAKDOWNPERCENT_PNG)
    api.update_status(
        status="Estimated split between WW2 and CW battles",
        media_ids=[battles.media_id, percent.media_id]
    )


def get_universal_params(config):
    params = dict()
    watermark = config.get('watermark text', None)
    if watermark:
        params['watermark_text'] = watermark
    return params


async def make_selection(config, args):
    additional_params = get_universal_params(config)
    now = datetime.utcnow()
    db = await connect(**config['database'])
    if args.top_cw_tanks or args.top_ww2_tanks:
        popular_tanks = await query_for_top_tanks(config, db)
    if args.activity_graphs:
        try:
            create_activity_graphs(*(await query_for_graphs(config, db)), **additional_params)
            if args.upload:
                upload_activity_graphs_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.account_age:
        try:
            create_account_age_chart(await query_for_active_accounts(config, db), **additional_params)
            if args.upload:
                upload_account_age_graph_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.accounts_by_battles:
        try:
            create_accounts_by_battles_chart(await query_for_accounts_by_battles(config, db), **additional_params)
            if args.upload:
                upload_accounts_by_battles_chart_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.five_battles_min:
        try:
            create_five_battles_minimum_chart(await query_five_battles_a_day_minimum(config, db), **additional_params)
            if args.upload:
                upload_five_battles_minimum_chart_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    # Limit long-term views to beginning of month to review previous month's history
    if args.long_term:
        if now.day == 1:
            try:
                create_long_term_charts(*await query_long_term_data(config, db, config.get('omit errors long term', True)), **additional_params)
                if args.mode_breakdown:
                    create_mode_difference_graph(*await query_for_mode_battles_difference(config, db, long_term=True), long_term=True, **additional_params)
                if args.upload:
                    upload_long_term_charts(config)
                    if args.mode_breakdown:
                        upload_long_term_mode_charts(config)
            except Exception as e:
                # print(e)
                traceback.print_exc()
    if args.share_unique:
        try:
            share_unique_with_twitter(config, await query_for_unique(config, db))
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.top_cw_tanks:
        try:
            share_top_tanks(config, 'CW', popular_tanks['CW'], (now - timedelta(days=1)).strftime('%Y-%m-%d'))
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.top_ww2_tanks:
        try:
            share_top_tanks(config, 'WW2', popular_tanks['WW2'], (now - timedelta(days=1)).strftime('%Y-%m-%d'))
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.mode_breakdown:
        try:
            create_mode_difference_graph(*await query_for_mode_battles_difference(config, db), **additional_params)
            if args.upload:
                upload_mode_breakdown_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()


if __name__ == '__main__':
    agp = ArgumentParser(description='Bot for processing tracker data and uploading to Twitter')
    agp.add_argument('config', help='Config file location')
    agp.add_argument('-u', '--upload', help='Upload to twitter', action='store_true')
    agp.add_argument('--activity-graphs', action='store_true')
    agp.add_argument('--account-age', action='store_true')
    agp.add_argument('--accounts-by-battles', action='store_true')
    agp.add_argument('--five-battles-min', action='store_true')
    agp.add_argument('--long-term', action='store_true')
    agp.add_argument('--share-unique', action='store_true')
    agp.add_argument('--top-cw-tanks', action='store_true')
    agp.add_argument('--top-ww2-tanks', action='store_true')
    agp.add_argument('--mode-breakdown', action='store_true')
    args = agp.parse_args()
    config = manage_config('read', args.config)
    if system() == 'Windows':
        from asyncio import set_event_loop_policy, WindowsSelectorEventLoopPolicy
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    run(make_selection(config, args))
