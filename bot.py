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
                    "filters": {
                        "filters": {
                            "xbox": {
                                "query_string": {
                                    "query": "account_id:<1073740000",
                                    "analyze_wildcard": True,
                                    "default_field": "*"
                                }
                            },
                            "ps": {
                                "query_string": {
                                    "query": "account_id:>=1073740000",
                                    "analyze_wildcard": True,
                                    "default_field": "*"
                                }
                            }
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
                  "filters": {
                      "filters": {
                          "xbox": {
                              "query_string": {
                                  "query": "account_id:<1073740000",
                                  "analyze_wildcard": True,
                                  "default_field": "*"
                                }
                          },
                          "ps": {
                              "query_string": {
                                  "query": "account_id:>=1073740000",
                                  "analyze_wildcard": True,
                                  "default_field": "*"
                              }
                          }
                      }
                  }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes":[]},
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

BATTLES_PNG = '/tmp/battles.png'
PLAYERS_PNG = '/tmp/players.png'

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
                    'es index': 'diff_battles-*'
                },
                f
            )

def query_es(config):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'])
    es = Elasticsearch(**config['elasticsearch'])
    # Setup queries
    battles_query['query']['bool']['must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    battles_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    players_query['query']['bool']['must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    players_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    # Query Elasticsearch
    battles = es.search(index=config['es index'], body=battles_query)
    players = es.search(index=config['es index'], body=players_query)
    # Filter numbers
    battles_xbox = [b['3']['buckets']['xbox']['1']['value'] for b in battles['aggregations']['2']['buckets']]
    battles_ps = [b['3']['buckets']['ps']['1']['value'] for b in battles['aggregations']['2']['buckets']]
    players_xbox = [b['3']['buckets']['xbox']['doc_count'] for b in players['aggregations']['2']['buckets']]
    players_ps = [b['3']['buckets']['ps']['doc_count'] for b in players['aggregations']['2']['buckets']]
    dates = [b['key_as_string'].split('T')[0] for b in players['aggregations']['2']['buckets']]
    return dates, battles_xbox, battles_ps, players_xbox, players_ps

def create_graphs(dates, battles_xbox, battles_ps, players_xbox, players_ps):
    # Players PNG
    plt.clf()
    plt.figure(figsize=(11,8), dpi=150)
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
    plt.figure(figsize=(11,8), dpi=150)
    plt.suptitle('Total Battles Per Platform')
    plt.xticks(rotation=45, ha='right')
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.plot(dates, battles_xbox, color='green', linewidth=2, label='Xbox')
    plt.plot(dates, battles_ps, color='blue', linewidth=2, label='Playstation')
    plt.grid()
    plt.legend()
    plt.savefig(BATTLES_PNG)

def upload_to_twitter(config):
    auth = OAuthHandler(config['twitter']['api key'], config['twitter']['api secret key'])
    auth.set_access_token(config['twitter']['access token'], config['twitter']['access token secret'])
    api = API(auth)
    battles = api.media_upload(BATTLES_PNG)
    players = api.media_upload(PLAYERS_PNG)
    api.update_status(
        status=config['twitter']['message'],
        media_ids=[players.media_id, battles.media_id]
    )

if __name__ == '__main__':
    agp = ArgumentParser(description='Bot for processing tracker data and uploading to Twitter')
    agp.add_argument('config', help='Config file location')
    args = agp.parse_args()
    config = manage_config('read', args.config)
    # dates, battles_xbox, battles_ps, players_xbox, players_ps = query_es(config)
    create_graphs(*query_es(config))
    upload_to_twitter(config)
