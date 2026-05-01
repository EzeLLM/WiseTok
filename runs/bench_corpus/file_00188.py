#!/usr/bin/env python3
"""API Status Monitor

Usage:
  statusmonitor.py produce
  statusmonitor.py consume

Options:
  -h, --help     Show this help.

Environment variables:

Environment variables allow separtion of configuration and source.
The sane defaults are provided for attaching to local Docker environment,
see `docker-compose.yml`

  logging:
  * LOG_LEVEL               (default: WARNING)

  consume:
  * KAFKA_BOOTSTRAP_SERVERS (default: localhost:9092)
  * KAFKA_STATUS_TOPIC      (default: statuses)
  * KAFKA_SSL_CA_FILE
  * KAFKA_SSL_CERT_FILE
  * KAFKA_SSL_KEY_FILE
  * PGHOST                  (default: localhost)
  * PGPORT                  (default: 5432)
  * PGSSLMODE               (default: None)
  * PGUSER                  (default: postgres)
  * PGPW                    (default: postgres)
  * PGDB                    (default: postgres)
  * PGAPPUSER               (default: postgres)
  * PGAPPPW                 (default: postgres)
  * PGAPPDB                 (default: api_statuses)

  produce:
  * KAFKA_BOOTSTRAP_SERVERS (default: localhost:9092)
  * KAFKA_STATUS_TOPIC      (default: statuses)
  * KAFKA_SSL_CA_FILE
  * KAFKA_SSL_CERT_FILE
  * KAFKA_SSL_KEY_FILE
  * API_READ_INTERVAL_SECS  (default: 5)
  * READER_CONF_FILE        (default: conf/apireader.yml)
"""


import logging
import os
import signal
import yaml
from docopt import docopt
from api_status_monitor.consumer.consumer import Consumer
from api_status_monitor.producer.producer import Producer


LOG_FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=LOG_FORMAT,
                    level=os.environ.get("LOG_LEVEL", "WARNING"))


def _get_ssl_configuration():
    """Return SSL configuration dictionary. Dictionary keys are aligned to
    kafka-python library expected keys.
    """
    ssl_ca_file = os.environ.get("KAFKA_SSL_CA_FILE", None)
    ssl_cert_file = os.environ.get("KAFKA_SSL_CERT_FILE", None)
    ssl_key_file = os.environ.get("KAFKA_SSL_KEY_FILE", None)
    if ssl_ca_file and ssl_cert_file and ssl_key_file:
        logging.info("SSL enabled for Kafka connection.")
        return {
            "ssl_cafile": ssl_ca_file,
            "ssl_certfile": ssl_cert_file,
            "ssl_keyfile": ssl_key_file
            }
    return None


def run_consumer():
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS",
                                       "localhost:9092")
    status_topic = os.environ.get("KAFKA_STATUS_TOPIC", "statuses")
    pghost = os.environ.get("PGHOST", "localhost")
    pgport = os.environ.get("PGPORT", "5432")
    pgsslmode = os.environ.get("PGSSLMODE", None)
    pguser = os.environ.get("PGUSER", "postgres")
    pgpw = os.environ.get("PGPW", "postgres")
    pgappuser = os.environ.get("PGAPPUSER", "postgres")
    pgapppw = os.environ.get("PGAPPPW", "postgres")
    pgdb = os.environ.get("PGDB", "postgres")
    appdb = os.environ.get("APPDB", "api_statuses")

    ssl = _get_ssl_configuration()

    consumer = Consumer(bootstrap_servers, status_topic, ssl)

    # Set the SIGTERM handler for terminating by orchestration environment
    signal.signal(signal.SIGTERM, consumer.sig_handler)
    # Set SIGINT handler for manual testing, Ctrl+C
    signal.signal(signal.SIGINT, consumer.sig_handler)

    # In production deployment the administrative actions and application
    # database access must be separated to different users with
    # different privileges.
    consumer.migrate(pghost, pgport, pguser, pgpw, pgdb, appdb, pgsslmode)
    consumer.connect_db(pghost, pgport, pgappuser, pgapppw, appdb, pgsslmode)
    consumer.run()


def run_producer():
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS",
                                       "localhost:9092")
    status_topic = os.environ.get("KAFKA_STATUS_TOPIC", "statuses")
    api_read_interval_secs = os.environ.get("API_READ_INTERVAL_SECS", 5)
    ssl = _get_ssl_configuration()

    reader_conf_file = os.environ.get("READER_CONF_FILE", "./conf/apireader.yml")
    with open(reader_conf_file) as f:
        reader_conf_dict = yaml.load(f, Loader=yaml.FullLoader)

    producer = Producer(reader_conf_dict, bootstrap_servers, status_topic, ssl)

    # Set the SIGTERM handler for terminating by orchestration environment
    signal.signal(signal.SIGTERM, producer.sig_handler)
    # Set SIGINT handler for manual testing, Ctrl+C
    signal.signal(signal.SIGINT, producer.sig_handler)

    producer.create_topic()
    producer.produce(api_read_interval_secs)


def main():
    arguments = docopt(__doc__)
    if arguments.get("consume"):
        run_consumer()
    elif arguments.get("produce"):
        run_producer()
    else:
        logging.warning("Unknown command")

    logging.info("Exiting.")


if __name__ == "__main__":
    main()
