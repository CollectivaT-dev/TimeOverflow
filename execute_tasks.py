# 2020 Col·lectivaT
#
# Top level script to execute the periodic queries
#
import os
import db
from datetime import datetime

def main(psql_config):
    date_config = datetime(2013,1,1), datetime(2020,1,1), datetime(2019,9,1)
    to_db = db.DB(config = psql_config,
                  date_config = date_config)
    to_db.get_active_users()
    print(to_db.active_users)

if __name__=="__main__":
    psql_config = (os.environ.get('TO_DB_SERVER'),
                   os.environ.get('TO_DB_USER'),
                   os.environ.get('TO_DB_PASSWORD'),
                   os.environ.get('TO_DB_NAME'))
    for element in psql_config:
        if not element:
            raise ValueError('TO_DB_SERVER, TO_DB_USER, TO_DB_PASSWORD and TO_DB_NAME '\
                             'has to be set as environment variables.')
    main(psql_config)
