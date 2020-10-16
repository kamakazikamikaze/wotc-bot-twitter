from argparse import ArgumentParser
from datetime import datetime, timedelta
from elasticsearch6 import Elasticsearch
from json import dump, load
from matplotlib import pyplot as plt
from tweepy import OAuthHandler, API

battles_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "date",
                "interval": "1d",
                "time_zone": "America/Chicago",
                "min_doc_count": 1
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "1": "desc"
                        }
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
                "time_zone": "America/Chicago",
                "min_doc_count": 1
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "_count": "desc"
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
                "time_zone": "America/Chicago",
                "min_doc_count": 1
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "_count": "desc"
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
                            "gte": "now-14d/d",
                            "lt": "now"
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

BATTLES_PNG = '/tmp/battles.png'
PLAYERS_PNG = '/tmp/players.png'
NEWPLAYERS_PNG = '/tmp/newplayers.png'

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
                        'message': "Today's update on the unique player count and total battles per platform for #WoTConsole."
                    },
                    'elasticsearch': {
                        'hosts': ['127.0.0.1']
                    },
                    'es index': 'diff_battles-*',
                    'unique': [7, 14, 30]
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
        'must'][-1]['range']['created_at']['lte'] = now.strftime('%Y-%m-%d')
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
    return dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_xbox, newplayers_ps


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


def create_graphs(dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_xbox, newplayers_ps):
    # Players PNG
    plt.clf()
    plt.figure(figsize=(11, 8), dpi=150)
    plt.suptitle('Unique Players Per Platform')
    plt.xticks(rotation=45, ha='right')
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.plot(dates, players_xbox, color='green', linewidth=2, label='Xbox')
    plt.plot(dates, players_ps, color='blue', linewidth=2, label='Playstation')
    plt.grid()
    plt.legend()
    plt.savefig(PLAYERS_PNG)
    # Battles PNG
    plt.clf()
    plt.figure(figsize=(11, 8), dpi=150)
    plt.suptitle('Total Battles Per Platform')
    plt.xticks(rotation=45, ha='right')
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.plot(dates, battles_xbox, color='green', linewidth=2, label='Xbox')
    plt.plot(dates, battles_ps, color='blue', linewidth=2, label='Playstation')
    plt.grid()
    plt.legend()
    plt.savefig(BATTLES_PNG)
    # New Players PNG
    plt.clf()
    plt.figure(figsize=(11, 8), dpi=150)
    plt.suptitle('New Players Per Platform')
    plt.xticks(rotation=45, ha='right')
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.plot(dates, newplayers_xbox, color='green', linewidth=2, label='Xbox')
    plt.plot(dates, newplayers_ps, color='blue', linewidth=2, label='Playstation')
    plt.grid()
    plt.legend()
    plt.savefig(NEWPLAYERS_PNG)


def upload_to_twitter(config):
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
    # dates, battles_xbox, battles_ps, players_xbox, players_ps = query_es(config)
    create_graphs(*query_es_for_graphs(config))
    upload_to_twitter(config)
    share_unique_with_twitter(config, query_es_for_unique(config))
