from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime, timedelta
from elasticsearch6 import Elasticsearch
from json import dump, load
from math import pi, sin, cos
from matplotlib import pyplot as plt
from tweepy import OAuthHandler, API


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
                'time_zone': 'America/Chicago',
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


BATTLES_PNG = '/tmp/battles.png'
PLAYERS_PNG = '/tmp/players.png'
NEWPLAYERS_PNG = '/tmp/newplayers.png'
ACCOUNTAGE_PNG = '/tmp/accountage.png'

def manage_config(mode, filename='config.json'):
    if mode == 'read':
        with open(filename) as f:
            return load(f)
    elif mode == 'create':
        with open(filename, 'w') as f:
            dump(
                {
                    'days': 14,
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
                    'account age': [7, 30, 90, 180, 365, 730, 1095, 1460, 1825]
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
    dates = [b['key_as_string'].split('T')[0] for b in players[
        'aggregations']['2']['buckets']]
    newplayers_dates = [b['key_as_string'].split('T')[0] for b in newplayers[
        'aggregations']['2']['buckets']]
    return dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_dates, newplayers_xbox, newplayers_ps


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


def create_activity_graphs(dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_dates, newplayers_xbox, newplayers_ps):
    # Players PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    plt.suptitle('Active Players Per Platform')
    plt.xticks(rotation=45, ha='right')
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.plot(dates, players_xbox, color='green', linewidth=2, label='Xbox')
    plt.plot(dates, players_ps, color='blue', linewidth=2, label='Playstation')
    plt.grid()
    plt.legend()
    plt.savefig(PLAYERS_PNG)
    del fig
    # Battles PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    plt.suptitle('Total Battles Per Platform')
    plt.xticks(rotation=45, ha='right')
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.plot(dates, battles_xbox, color='green', linewidth=2, label='Xbox')
    plt.plot(dates, battles_ps, color='blue', linewidth=2, label='Playstation')
    plt.grid()
    plt.legend()
    plt.savefig(BATTLES_PNG)
    del fig
    # New Players PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    plt.suptitle('New Players Per Platform')
    plt.xticks(rotation=45, ha='right')
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.plot(newplayers_dates, newplayers_xbox, color='green', linewidth=2, label='Xbox')
    plt.plot(newplayers_dates, newplayers_ps, color='blue', linewidth=2, label='Playstation')
    plt.grid()
    plt.legend()
    plt.savefig(NEWPLAYERS_PNG)
    del fig


def query_es_for_chart(config):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    es = Elasticsearch(**config['elasticsearch'])
    personal_players_query['query']['bool']['must'][-1]['range']['date']['gt'] = then.strftime('%Y-%m-%d')
    personal_players_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')

    # Get all account IDs of active players
    hits = []
    response = es.search(index='total_battles-*', body=personal_players_query, scroll='30s')
    while len(response['hits']['hits']):
        hits.extend(response['hits']['hits'])
        response = es.scroll(response['_scroll_id'], scroll='3s')

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


def create_account_age_chart(buckets):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    plt.suptitle("Breakdown of yesterday's active accounts by account age")
    ax = plt.axes()
    ax.axis('equal')
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
    outer_wedges, outer_text, outer_autotext = ax.pie(
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
        ax.annotate(
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
    inner_wedges, inner_text, inner_autotext = ax.pie(
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

    ax.legend(inner_wedges[-2:], ['xbox', 'ps'], loc='lower right')
    plt.text(0.5, 0.5, '@WOTC_Tracker', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.savefig(ACCOUNTAGE_PNG)
    del fig


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
    api.update_status(
        status=config['twitter']['message'],
        media_ids=[players.media_id, battles.media_id, newplayers.media_id]
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


def share_unique_with_twitter(config, unique):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    status = 'Unique Player Count For {} Over Time\n{}'
    formatting = '{} days: {}'
    for key, values in unique.items():
        api.update_status(
            status=status.format(
                key,
                '\n'.join(map(lambda l: formatting.format(
                    config['unique'][values.index(l)], l), values))
            )
        )

if __name__ == '__main__':
    agp = ArgumentParser(
        description='Bot for processing tracker data and uploading to Twitter')
    agp.add_argument('config', help='Config file location')
    args = agp.parse_args()
    config = manage_config('read', args.config)
    create_activity_graphs(*query_es_for_graphs(config))
    upload_activity_graphs_to_twitter(config)
    create_account_age_chart(query_es_for_chart(config))
    upload_account_age_graph_to_twitter(config)
    share_unique_with_twitter(config, query_es_for_unique(config))
