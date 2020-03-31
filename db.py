# 2020 Col·lectivaT
#
# DB class for queries
#

import psycopg2
import queries
import pandas
from datetime import datetime

class DB(object):
    def __init__(self, config, date_config):
        server, to_user, to_password, to_database = config
        self.connection = psycopg2.connect(host=server,
                                           database=to_database,
                                           user=to_user,
                                           password=to_password)
        self.date_start, self.date_end, self.date_active_member = date_config

        self.active_users = None

    def get_active_users(self, organization_id=None):
        active_date_start = min(self.date_start,
                                self.date_active_member)
        active_users_query = queries.ACTIVE_USERS%(
                             datetime.strftime(active_date_start, '%Y-%m-%d'),
                             datetime.strftime(self.date_end, '%Y-%m-%d'))
        print(active_users_query)

        if organization_id == None:
            # get the whole active users in order not to query multiple times
            if not self.active_users:
                self.active_users = pandas.read_sql_query(active_users_query,
                                                          self.connection)
        else:
            if not self.active_users:
                # get the whole active users
                self.get_active_users()
