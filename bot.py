from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime, timedelta
from elasticsearch6 import Elasticsearch
from json import dump, load
from math import pi, sin, cos
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from tweepy import OAuthHandler, API


# Multi-day, use gte
battles_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "date",
                "interval": "1d",
                "min_doc_count": 0
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "1": "desc"
                        },
                        "min_doc_count": 0
                    },
                    "aggs": {
                        "1": {
                            "sum": {
                                "field": "battles"
                            }
                        }
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

# Multi-day, use gte
players_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "date",
                "interval": "1d",
                "min_doc_count": 0
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "_count": "desc"
                        },
                        "min_doc_count": 0
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

unique_count_query = {
    "aggs": {
        "2": {
            "terms": {
                "field": "console.keyword",
                "size": 2,
                "order": {
                    "1": "desc"
                }
            },
            "aggs": {
                "1": {
                    "cardinality": {
                        "field": "account_id"
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

new_players_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "created_at",
                "interval": "1d",
                "min_doc_count": 0
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "_count": "desc"
                        },
                        "min_doc_count": 0
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "created_at",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "created_at": {
                            "gte": None,
                            "lt": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

personal_players_query = {
    'sort': [],
    '_source': {'excludes': []},
    'aggs': {
        '2': {
            'date_histogram': {
                'field': 'date',
                'interval': '1d',
                'min_doc_count': 0
            }
        }
    },
    'stored_fields': ['_source'],
    'script_fields': {},
    'docvalue_fields': [{'field': 'date', 'format': 'date_time'}],
    'query': {
        'bool': {
            'must': [
                {'match_all': {}},
                {
                    'range': {
                        'date': {
                            'gt': None,
                            'lte': None,
                            'format': 'date'
                        }
                    }
                }
            ],
           'filter': [],
           'should': [],
           'must_not': []
        }
    },
    'size': 500
}

accounts_per_battles_range_query = {
    'aggs': {
        '2': {
            'range': {
                'field': 'battles',
                'ranges': [
                    {'from': 1, 'to': 5},
                    {'from': 5, 'to': 10},
                    {'from': 10, 'to': 20},
                    {'from': 20, 'to': 30},
                    {'from': 30, 'to': 40},
                    {'from': 40, 'to': 50},
                    {'from': 50}
                ],
                'keyed': True
            },
            'aggs': {
                '3': {
                    'terms': {
                        'field': 'console.keyword',
                        'size': 2,
                        'order': {'_count': 'desc'}
                    }
                }
            }
        }
    },
    'size': 0,
    '_source': {'excludes': []},
    'stored_fields': ['*'],
    'script_fields': {},
    'docvalue_fields': [{'field': 'date', 'format': 'date_time'}],
    'query': {
        'bool': {
            'must': [
                {'match_all': {}},
                {'match_all': {}},
                {'range': {'date': {'gt': None, 'lte': None, 'format': 'date'}}}
            ],
            'filter': [],
            'should': [],
            'must_not': []
        }
    }
}

five_battles_a_day_query = {
    'aggs': {
        '4': {
            'date_histogram': {
                'field': 'date',
                'interval': '1d',
                'min_doc_count': 0
            },
            'aggs': {
                '3': {
                    'terms': {
                        'field': 'console.keyword',
                        'size': 2,
                        'order': {'_count': 'desc'}
                    },
                    'aggs': {
                        '2': {
                            'range': {
                                'field': 'battles',
                                'ranges': [{'from': 5, 'to': None}],
                                'keyed': True
                            }
                        }
                    }
                }
            }
        }
    },
    'size': 0,
    '_source': {'excludes': []},
    'stored_fields': ['*'],
    'script_fields': {},
    'docvalue_fields': [{'field': 'date', 'format': 'date_time'}],
    'query': {
        'bool': {
            'must': [
                {'match_all': {}},
                {'match_all': {}},
                {
                    'range': {
                        'date': {
                            'gte': None,
                            'lte': None,
                            'format': 'date'
                        }
                    }
                }
            ],
            'filter': [],
            'should': [],
            'must_not': []
        }
    }
}

BATTLES_PNG = '/tmp/battles.png'
PLAYERS_PNG = '/tmp/players.png'
NEWPLAYERS_PNG = '/tmp/newplayers.png'
AVERAGE_PNG = '/tmp/average.png'
ACCOUNTAGE_PNG = '/tmp/accountage.png'
BATTLERANGE_PNG = '/tmp/battlerange.png'
FIVEADAY_PNG = '/tmp/fiveaday.png'
PLAYERSLONG_PNG = '/tmp/playerslong.png'
BATTLESLONG_PNG = '/tmp/battleslong.png'
AVERAGELONG_PNG = '/tmp/averagelong.png'

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
                    'elasticsearch': {
                        'hosts': ['127.0.0.1']
                    },
                    'es index': 'diff_battles-*',
                    'unique': [7, 14, 30],
                    'account age': [7, 30, 90, 180, 365, 730, 1095, 1460, 1825],
                    'battle ranges': [
                        {"from": 1, "to": 5},
                        {"from": 5, "to": 10},
                        {"from": 10, "to": 20},
                        {"from": 20, "to": 30},
                        {"from": 30, "to": 40},
                        {"from": 40, "to": 50},
                        {"from": 50}
                    ],
                    'watermark text': '@WOTC_Tracker'
                }
            )


def query_es_for_graphs(config):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'])
    es = Elasticsearch(**config['elasticsearch'])
    # Setup queries
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    new_players_query['query']['bool'][
        'must'][-1]['range']['created_at']['gte'] = then.strftime('%Y-%m-%d')
    new_players_query['query']['bool'][
        'must'][-1]['range']['created_at']['lt'] = now.strftime('%Y-%m-%d')
    # Query Elasticsearch
    battles = es.search(index=config['es index'], body=battles_query)
    players = es.search(index=config['es index'], body=players_query)
    newplayers = es.search(index='players', body=new_players_query)
    # Filter numbers
    battles_xbox = []
    battles_ps = []
    players_xbox = []
    players_ps = []
    newplayers_xbox = []
    newplayers_ps = []
    averages_xbox = []
    averages_ps = []
    for bucket in battles['aggregations']['2']['buckets']:
        if not bucket['3']['buckets']:
            battles_xbox.append(0)
            battles_ps.append(0)
            continue
        for subbucket in bucket['3']['buckets']:
            if subbucket['key'] == 'xbox':
                battles_xbox.append(subbucket['1']['value'])
            else:
                battles_ps.append(subbucket['1']['value'])
    for bucket in players['aggregations']['2']['buckets']:
        if not bucket['3']['buckets']:
            players_xbox.append(0)
            players_ps.append(0)
            continue
        for subbucket in bucket['3']['buckets']:
            if subbucket['key'] == 'xbox':
                players_xbox.append(subbucket['doc_count'])
            else:
                players_ps.append(subbucket['doc_count'])
    for bucket in newplayers['aggregations']['2']['buckets']:
        if not bucket['3']['buckets']:
            newplayers_xbox.append(0)
            newplayers_ps.append(0)
        for subbucket in bucket['3']['buckets']:
            if subbucket['key'] == 'xbox':
                newplayers_xbox.append(subbucket['doc_count'])
            else:
                newplayers_ps.append(subbucket['doc_count'])
    for b, p in zip(battles_xbox, players_xbox):
        averages_xbox.append(b / p)
    for b, p in zip(battles_ps, players_ps):
        averages_ps.append(b / p)
    dates = [b['key_as_string'].split('T')[0] for b in players[
        'aggregations']['2']['buckets']]
    newplayers_dates = [b['key_as_string'].split('T')[0] for b in newplayers[
        'aggregations']['2']['buckets']]
    return dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_dates, newplayers_xbox, newplayers_ps, averages_xbox, averages_ps


def query_es_for_unique(config):
    now = datetime.utcnow()
    es = Elasticsearch(**config['elasticsearch'])
    unique = {'Xbox': [], 'Playstation': []}
    unique_count_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    for earliest in config['unique']:
        unique_count_query['query']['bool']['must'][-1]['range']['date'][
            'gte'] = (now - timedelta(days=earliest)).strftime('%Y-%m-%d')
        results = es.search(index=config['es index'], body=unique_count_query)
        for bucket in results['aggregations']['2']['buckets']:
            if bucket['key'] == 'xbox':
                unique['Xbox'].append(bucket['1']['value'])
            else:
                unique['Playstation'].append(bucket['1']['value'])
    return unique


def create_activity_graphs(dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_dates, newplayers_xbox, newplayers_ps, averages_xbox, averages_ps, watermark_text='@WOTC_Tracker'):
    shifted_dates = [(datetime.strptime(d, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d') for d in dates]
    # Players PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('Active Accounts Per Platform')
    # ax1 = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, players_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(shifted_dates, players_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
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
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, battles_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(shifted_dates, battles_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
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
    ax1.set_xticklabels(dates, ha='right')
    ax1.plot(newplayers_dates, newplayers_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(newplayers_dates, newplayers_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
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
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, averages_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(shifted_dates, averages_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(AVERAGE_PNG)
    del fig


def query_es_for_active_accounts(config):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    es = Elasticsearch(**config['elasticsearch'])
    personal_players_query['query']['bool']['must'][-1]['range']['date']['gt'] = then.strftime('%Y-%m-%d')
    personal_players_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')

    # Get all account IDs of active players
    hits = []
    response = es.search(index=config['es index'], body=personal_players_query, scroll='30s')
    while len(response['hits']['hits']):
        hits.extend(response['hits']['hits'])
        response = es.scroll(scroll_id=response['_scroll_id'], scroll='3s')

    flattened = [doc['_source']['account_id'] for doc in hits]

    # Query account information to get age details
    player_info_extracted = []
    for i in range(0, len(flattened), 10000):
        active_player_info = es.mget(index='players', doc_type='player', body={'ids': flattened[i:i+10000]}, _source=['account_id', 'console', 'created_at'])
        player_info_extracted.extend([doc['_source'] for doc in active_player_info['docs']])

    sorted_player_info = sorted(player_info_extracted, key = lambda d: d['created_at'])
    buckets = {
        "xbox": OrderedDict((v, 0) for v in sorted(config['account age'])),
        "ps": OrderedDict((v, 0) for v in sorted(config['account age'])),
        "all": OrderedDict((v, 0) for v in sorted(config['account age']))
    }

    # Sum account ages based on range of age
    buckets['xbox']['other'] = 0
    buckets['ps']['other'] = 0
    buckets['all']['other'] = 0
    for player in sorted_player_info:
        delta = now - datetime.strptime(player['created_at'], '%Y-%m-%dT%H:%M:%S')
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


def query_es_for_accounts_by_battles(config):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    es = Elasticsearch(**config['elasticsearch'])
    accounts_per_battles_range_query['query']['bool']['must'][-1]['range']['date']['gt'] = then.strftime('%Y-%m-%d')
    accounts_per_battles_range_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    if 'battle ranges' in config:
        accounts_per_battles_range_query['aggs']['2']['range']['ranges'] = config['battle ranges']

    response = es.search(index=config['es index'], body=accounts_per_battles_range_query)
    buckets = {
        "xbox": OrderedDict((v, 0) for v in response['aggregations']['2']['buckets'].keys()),
        "ps": OrderedDict((v, 0) for v in response['aggregations']['2']['buckets'].keys()),
        "all": OrderedDict((v, 0) for v in response['aggregations']['2']['buckets'].keys()),
    }
    for key, value in response['aggregations']['2']['buckets'].items():
        buckets['all'][key] = value['doc_count']
        for bucket in value['3']['buckets']:
            buckets[bucket['key']][key] = bucket['doc_count']
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

    outer_labels = []
    prev = 0
    for key in buckets['all'].keys():
        parts = key.split('-')
        outer_labels.append('{}-{} battles'.format(int(float(parts[0])) if parts[0] != '*' else parts[0], int(float(parts[1])) - 1 if parts[1] != '*' else parts[1]))

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


def query_five_battles_a_day_minimum(config):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'])
    es = Elasticsearch(**config['elasticsearch'])
    five_battles_a_day_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    five_battles_a_day_query['query']['bool']['must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    response = es.search(index=config['es index'], body=five_battles_a_day_query)

    buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    for bucket in response['aggregations']['4']['buckets']:
        key = bucket['key_as_string'].split('T')[0]
        buckets['xbox'][key] = 0
        buckets['ps'][key] = 0
        buckets['all'][key] = 0
        for subbucket in bucket['3']['buckets']:
            buckets[subbucket['key']][key] = subbucket['2']['buckets']['5.0-*']['doc_count']
        buckets['all'][key] = buckets['xbox'][key] + buckets['ps'][key]

    return buckets


# Requested by Khorne Dog in the forums
def create_five_battles_minimum_chart(buckets, watermark_text='@WOTC_Tracker'):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle("Number of accounts having played at least 5 battles")
    ax1 = fig.add_subplot(111)

    width = 0.25
    keys = [(datetime.strptime(d, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d') for d in buckets['all'].keys()]
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
        colLabels=keys,
        loc='bottom')
    ax1.set_ylabel('Accounts')
    ax1.set_xticks([])
    ax1.legend((xbox_bars[0], ps_bars[0]), ('xbox', 'ps'))
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(FIVEADAY_PNG)


def query_long_term_data(config, filter_server_failures=True):
    now = datetime.utcnow()
    then = now - timedelta(days=config.get('long term', 90) + 1)
    es = Elasticsearch(**config['elasticsearch'])
    # Setup queries
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')

    players = es.search(index=config['es index'], body=players_query)
    battles = es.search(index=config['es index'], body=battles_query)

    players_buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    battles_buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    average_battles_per_day_buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    for bucket in players['aggregations']['2']['buckets']:
        key = bucket['key_as_string'].split('T')[0]
        players_buckets['xbox'][key] = 0
        players_buckets['ps'][key] = 0
        players_buckets['all'][key] = 0
        if not bucket['3']['buckets']:
            continue
        for subbucket in bucket['3']['buckets']:
            players_buckets[subbucket['key']][key] = subbucket['doc_count']
        players_buckets['all'][key] = players_buckets['xbox'][key] + players_buckets['ps'][key]

    for bucket in battles['aggregations']['2']['buckets']:
        key = bucket['key_as_string'].split('T')[0]
        battles_buckets['xbox'][key] = 0
        battles_buckets['ps'][key] = 0
        battles_buckets['all'][key] = 0
        if not bucket['3']['buckets']:
            continue
        for subbucket in bucket['3']['buckets']:
            battles_buckets[subbucket['key']][key] = subbucket['1']['value']
        battles_buckets['all'][key] = battles_buckets['xbox'][key] + battles_buckets['ps'][key]

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

    delkey = list(players_buckets['all'].keys())[0]
    # delkey = list(battles_buckets['all'].keys())[0]
    del players_buckets['all'][key]
    del players_buckets['xbox'][key]
    del players_buckets['ps'][key]
    del battles_buckets['all'][key]
    del battles_buckets['xbox'][key]
    del battles_buckets['ps'][key]
    del average_battles_per_day_buckets['xbox'][key]
    del average_battles_per_day_buckets['ps'][key]
    del average_battles_per_day_buckets['all'][key]

    return players_buckets, battles_buckets, average_battles_per_day_buckets


def create_long_term_charts(players_buckets, battles_buckets, average_battles_per_day_buckets, watermark_text='@WOTC_Tracker'):
    dates = [datetime.strptime(d, '%Y-%m-%d') - timedelta(days=1) for d in players_buckets['all'].keys()]
    # Players PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Active Accounts Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.plot(dates, players_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, players_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.set_xticks(dates)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.tight_layout()
    fig.autofmt_xdate()
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(PLAYERSLONG_PNG)
    del fig
    # Battles PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Total Battles Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.plot(dates, battles_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, battles_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.set_xticks(dates)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.tight_layout()
    fig.autofmt_xdate()
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(BATTLESLONG_PNG)
    del fig
    # Average PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Average Battles Played Per Account Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.plot(dates, average_battles_per_day_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, average_battles_per_day_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.set_xticks(dates)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.tight_layout()
    fig.autofmt_xdate()
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
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
    for key, values in unique.items():
        api.update_status(
            status=status.format(
                key,
                '\n'.join(map(lambda l: formatting.format(
                    config['unique'][values.index(l)], l), values))
            )
        )


def get_universal_params(config):
    params = dict()
    watermark = config.get('watermark text', None)
    if watermark:
        params['watermark_text'] = watermark
    return params


if __name__ == '__main__':
    agp = ArgumentParser(
        description='Bot for processing tracker data and uploading to Twitter')
    agp.add_argument('config', help='Config file location')
    args = agp.parse_args()
    config = manage_config('read', args.config)
    additional_params = get_universal_params(config)
    now = datetime.utcnow()
    try:
        create_activity_graphs(*query_es_for_graphs(config), **additional_params)
        upload_activity_graphs_to_twitter(config)
    except Exception as e:
        print(e)
    try:
        create_account_age_chart(query_es_for_active_accounts(config), **additional_params)
        upload_account_age_graph_to_twitter(config)
    except Exception as e:
        print(e)
    try:
        create_accounts_by_battles_chart(query_es_for_accounts_by_battles(config), **additional_params)
        upload_accounts_by_battles_chart_to_twitter(config)
    except Exception as e:
        print(e)
    try:
        create_five_battles_minimum_chart(query_five_battles_a_day_minimum(config), **additional_params)
        upload_five_battles_minimum_chart_to_twitter(config)
    except Exception as e:
        print(e)
    # Limit long-term views to beginning of month to review previous month's history
    if now.day == 1:
        try:
            create_long_term_charts(*query_long_term_data(config, config.get('omit errors long term', True)), **additional_params)
            upload_long_term_charts(config)
        except Exception as e:
            print(e)
    try:
        share_unique_with_twitter(config, query_es_for_unique(config))
    except Exception as e:
        print(e)
