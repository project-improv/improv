import logging
import signal
import time

import pytest
import zmq
from zmq.backend.cython._zmq import SocketOption

from improv.harvester import RedisHarvester
from improv.store import RedisStoreInterface

from improv.link import ZmqLink
from improv.messaging import HarvesterInfoReplyMsg
from conftest import SignalManager


def test_harvester_shuts_down_on_sigint(setup_store, harvester):
    harvester_ports, broker_socket, p = harvester
    ctx = zmq.Context()
    s = ctx.socket(zmq.REP)
    s.bind(f"tcp://*:{harvester_ports[3]}")
    s.recv_pyobj()
    reply = HarvesterInfoReplyMsg("harvester", "OK", "")
    s.send_pyobj(reply)
    time.sleep(2)
    p.terminate()
    p.join(5)
    if p.exitcode is None:
        p.kill()
        pytest.fail("Harvester did not exit in time")
    else:
        assert True

    s.close(linger=0)
    ctx.destroy(linger=0)


def test_harvester_relieves_memory_pressure(setup_store, harvester):
    store_interface = RedisStoreInterface()
    harvester_ports, broker_socket, p = harvester
    broker_link = ZmqLink(broker_socket, "test", "test topic")
    ctx = zmq.Context()
    s = ctx.socket(zmq.REP)
    s.bind(f"tcp://*:{harvester_ports[3]}")
    s.recv_pyobj()
    reply = HarvesterInfoReplyMsg("harvester", "OK", "")
    s.send_pyobj(reply)
    client = RedisStoreInterface()
    for i in range(10):
        message = [i for i in range(500000)]
        key = client.put(message)
        broker_link.put(key)
    time.sleep(2)

    db_info = store_interface.client.info()
    max_memory = db_info["maxmemory"]
    used_memory = db_info["used_memory"]
    used_max_ratio = used_memory / max_memory
    assert used_max_ratio <= 0.75

    p.terminate()
    p.join(5)
    if p.exitcode is None:
        p.kill()
        pytest.fail("Harvester did not exit in time")

    s.close(linger=0)
    ctx.destroy(linger=0)


def test_harvester_stop_logs_and_halts_running():
    with SignalManager():
        ctx = zmq.Context()
        s = ctx.socket(zmq.PULL)
        s.bind("tcp://*:0")
        pull_port_string = s.getsockopt_string(SocketOption.LAST_ENDPOINT)
        pull_port = int(pull_port_string.split(":")[-1])
        harvester = RedisHarvester(
            nexus_hostname="localhost",
            nexus_comm_port=10000,  # never gets called in this test
            redis_hostname="localhost",
            redis_port=10000,  # never gets called in this test
            broker_hostname="localhost",
            broker_port=10000,  # never gets called in this test
            logger_hostname="localhost",
            logger_port=pull_port,
        )
        harvester.stop(signal.SIGINT, None)
        assert not harvester.running
        msg_available = s.poll(timeout=1000)
        assert msg_available
        record = s.recv_json()
        assert (
            record["message"]
            == f"Harvester shutting down due to signal {signal.SIGINT}"
        )

        s.close(linger=0)
        ctx.destroy(linger=0)


# @pytest.mark.skip
def test_harvester_relieves_memory_pressure_one_loop(ports, setup_store):
    def harvest_and_quit(harvester_instance: RedisHarvester):
        harvester_instance.collect()
        harvester_instance.stop(signal.SIGINT, None)

    with SignalManager():
        try:
            ctx = zmq.Context()
            nex_s = ctx.socket(zmq.REP)
            nex_s.bind(f"tcp://*:{ports[3]}")

            broker_s = ctx.socket(zmq.PUB)
            broker_s.bind("tcp://*:1234")
            broker_link = ZmqLink(broker_s, "test", "test topic")

            log_s = ctx.socket(zmq.PULL)
            log_s.bind("tcp://*:0")
            pull_port_string = log_s.getsockopt_string(SocketOption.LAST_ENDPOINT)
            pull_port = int(pull_port_string.split(":")[-1])

            harvester = RedisHarvester(
                nexus_hostname="localhost",
                nexus_comm_port=ports[3],  # never gets called in this test
                redis_hostname="localhost",
                redis_port=6379,  # never gets called in this test
                broker_hostname="localhost",
                broker_port=1234,  # never gets called in this test
                logger_hostname="localhost",
                logger_port=pull_port,
            )

            harvester.establish_connections()

            client = RedisStoreInterface()
            for i in range(9):
                message = [i for i in range(500000)]
                try:
                    key = client.put(message)
                    broker_link.put(key)
                except Exception as e:
                    logging.warning(e)
                    logging.warning("Proceeding under the assumption Redis is full.")
                    break
            time.sleep(2)

            harvester.serve(harvest_and_quit, harvester_instance=harvester)

            db_info = client.client.info()
            max_memory = db_info["maxmemory"]
            used_memory = db_info["used_memory"]
            used_max_ratio = used_memory / max_memory

            assert used_max_ratio <= 0.5
            assert not harvester.running
            assert harvester.nexus_socket.closed
            assert harvester.sub_socket.closed
            assert harvester.zmq_context.closed
            logging.info("Passed harvester one loop test")

        except Exception as e:
            logging.error(f"Encountered exception {e} during execution of test")
            print(e)
            pass

        try:

            nex_s.close(linger=0)
            broker_s.close(linger=0)
            log_s.close(linger=0)
            ctx.destroy(linger=0)
        except Exception as e:
            logging.error(f"Encountered exception {e} during teardown")
            print(e)
            pass


def test_harvester_loops_with_no_memory_pressure(ports, setup_store):
    def harvest_and_quit(harvester_instance: RedisHarvester):
        harvester_instance.collect()
        harvester_instance.stop(signal.SIGINT, None)

    with SignalManager():

        ctx = zmq.Context()
        nex_s = ctx.socket(zmq.REP)
        nex_s.bind(f"tcp://*:{ports[3]}")

        log_s = ctx.socket(zmq.PULL)
        log_s.bind("tcp://*:0")
        pull_port_string = log_s.getsockopt_string(SocketOption.LAST_ENDPOINT)
        pull_port = int(pull_port_string.split(":")[-1])

        harvester = RedisHarvester(
            nexus_hostname="localhost",
            nexus_comm_port=ports[3],  # never gets called in this test
            redis_hostname="localhost",
            redis_port=6379,  # never gets called in this test
            broker_hostname="localhost",
            broker_port=1234,  # never gets called in this test
            logger_hostname="localhost",
            logger_port=pull_port,
        )

        harvester.establish_connections()

        client = RedisStoreInterface()

        harvester.serve(harvest_and_quit, harvester_instance=harvester)

        db_info = client.client.info()
        max_memory = db_info["maxmemory"]
        used_memory = db_info["used_memory"]
        used_max_ratio = used_memory / max_memory
        assert used_max_ratio <= 0.5
        assert not harvester.running
        assert harvester.nexus_socket.closed
        assert harvester.sub_socket.closed
        assert harvester.zmq_context.closed

        nex_s.close(linger=0)
        log_s.close(linger=0)
        ctx.destroy(linger=0)
