from argparse import ArgumentParser
from asyncio import run
from asyncpg import connect
from json import load
from requests import get

ENCYCLOPEDIA = 'https://api-console.worldoftanks.com/wotx/encyclopedia/vehicles/'

def query_tanks_from_api(api_key):
    return get(
        ENCYCLOPEDIA,
        params={
            'application_id': api_key,
            'fields': (
                'name,'
                'short_name,'
                'tank_id,'
                'is_premium,'
                'nation,'
                'era,'
                'tier,'
                'type'
            )
        }
    ).json()['data']


def tank_unwrapper(tanks):
    for tank in tanks:
        yield (
            tank['tank_id'], tank['name'], tank['short_name'], tank['is_premium'],
            tank['nation'], tank['era'], tank['tier'], tank['type']
        )


async def sync_database(config, tanks):
    db = await connect(**config['database'])
    __ = await db.execute('''
        CREATE TABLE IF NOT EXISTS tanks (
            tank_id INT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            short_name TEXT NOT NULL,
            is_premium BOOLEAN NOT NULL,
            nation TEXT,
            era TEXT,
            tier INT NOT NULL,
            type TEXT NOT NULL
        )
        ''')
    __ = await db.executemany('''
        INSERT INTO tanks (
            tank_id, name, short_name, is_premium, nation, era, tier, type
        ) VALUES (
            $1::int,
            $2::text,
            $3::text,
            $4::boolean,
            $5::text,
            $6::text,
            $7::int,
            $8::text
        ) ON CONFLICT DO NOTHING
        ''', tank_unwrapper(tanks.values()))


if __name__ == '__main__':
    agp = ArgumentParser()
    agp.add_argument('config')
    args = agp.parse_args()
    with open(args.config) as f:
        config = load(f)
    run(sync_database(config, query_tanks_from_api(config['wg api key'])))
