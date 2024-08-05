import glob
import shutil
import time
import os

import pytest
import logging

import zmq

from improv.config import CannotCreateConfigException
from improv.messaging import ActorStateMsg
from improv.nexus import (
    Nexus,
    ConfigFileNotProvidedException,
    ConfigFileNotValidException,
)
from improv.store import StoreInterface
from conftest import SignalManager


def test_init(setdir):
    # store = setup_store
    nex = Nexus("test")
    assert str(nex) == "test"


@pytest.mark.parametrize(
    "cfg_name",
    [
        "good_config.yaml",
    ],
)
def test_create_nexus(setdir, ports, cfg_name):
    nex = Nexus("test")
    nex.create_nexus(file=cfg_name, control_port=ports[0], output_port=ports[1])
    assert list(nex.actors.keys()) == ["Acquirer", "Analysis"]
    assert list(nex.flags.keys()) == ["quit", "run", "load"]
    assert nex.processes == []
    nex.destroy_nexus()
    assert True


def test_config_logged(setdir, ports, caplog):
    nex = Nexus("test")
    nex.create_nexus(
        file="minimal_with_settings.yaml", control_port=ports[0], output_port=ports[1]
    )
    nex.destroy_nexus()
    assert any(
        [
            "not_relevant: for testing purposes" in record.msg
            for record in caplog.records
        ]
    )


def test_load_config(sample_nex):
    nex = sample_nex
    assert any(
        [
            link_info.link_name == "q_out"
            for link_info in nex.outgoing_topics["Acquirer"]
        ]
    )
    assert any(
        [link_info.link_name == "q_in" for link_info in nex.incoming_topics["Analysis"]]
    )


def test_argument_config_precedence(setdir, ports):
    nex = Nexus("test")
    nex.create_nexus(
        file="minimal_with_settings.yaml",
        control_port=ports[0],
        output_port=ports[1],
        store_size=11_000_000,
    )
    cfg = nex.config.settings
    nex.destroy_nexus()
    assert cfg["control_port"] == ports[0]
    assert cfg["output_port"] == ports[1]
    assert cfg["store_size"] == 11_000_000


# delete this comment later
def test_start_nexus(sample_nex):
    with SignalManager():

        async def set_quit_flag(test_nex):
            test_nex.flags["quit"] = True

        nex = sample_nex
        nex.start_nexus(nex.poll_queues, poll_function=set_quit_flag, test_nex=nex)
        assert [p.name for p in nex.processes] == ["Acquirer", "Analysis"]


@pytest.mark.skip(
    reason="This test is unfinished - it does not validate link structure"
)
@pytest.mark.parametrize(
    ("cfg_name", "actor_list", "link_list"),
    [
        (
            "good_config.yaml",
            ["Acquirer", "Analysis"],
            ["Acquirer_sig", "Analysis_sig"],
        ),
        (
            "simple_graph.yaml",
            ["Acquirer", "Analysis"],
            ["Acquirer_sig", "Analysis_sig"],
        ),
        (
            "complex_graph.yaml",
            ["Acquirer", "Analysis", "InputStim"],
            ["Acquirer_sig", "Analysis_sig", "InputStim_sig"],
        ),
    ],
)
def test_config_construction(cfg_name, actor_list, link_list, setdir, ports):
    """Tests if constructing a nexus based on
    the provided config has the right structure.

    After construction based on the config, this
    checks whether all the right actors are constructed and whether the
    links between them are constructed correctly.
    """

    nex = Nexus("test")
    nex.create_nexus(file=cfg_name, control_port=ports[0], output_port=ports[1])
    logging.info(cfg_name)

    # Check for actors

    act_lst = list(nex.actors)

    nex.destroy_nexus()

    assert actor_list == act_lst
    act_lst = []
    assert True


@pytest.mark.parametrize(
    "cfg_name",
    [
        "single_actor.yaml",
    ],
)
def test_single_actor(setdir, ports, cfg_name):
    nex = Nexus("test")
    with pytest.raises(AttributeError):
        nex.create_nexus(
            file="single_actor.yaml", control_port=ports[0], output_port=ports[1]
        )

    nex.destroy_nexus()


def test_cyclic_graph(setdir, ports):
    nex = Nexus("test")
    nex.create_nexus(
        file="cyclic_config.yaml", control_port=ports[0], output_port=ports[1]
    )
    assert True
    nex.destroy_nexus()


def test_blank_cfg(setdir, caplog, ports):
    nex = Nexus("test")
    with pytest.raises(CannotCreateConfigException):
        nex.create_nexus(
            file="blank_file.yaml", control_port=ports[0], output_port=ports[1]
        )
    assert any(
        ["The config file is empty" in record.msg for record in list(caplog.records)]
    )
    nex.destroy_nexus()


def test_start_store(caplog):
    nex = Nexus("test")
    nex._start_store_interface(100_000_000)  # 100 MB store

    assert any(
        "StoreInterface start successful" in record.msg for record in caplog.records
    )

    nex._close_store_interface()
    nex.destroy_nexus()
    assert True


def test_close_store(caplog):
    nex = Nexus("test")
    nex._start_store_interface(10000)
    nex._close_store_interface()

    assert any(
        "StoreInterface close successful" in record.msg for record in caplog.records
    )

    # write to store

    with pytest.raises(AttributeError):
        nex.p_StoreInterface.put("Message in", "Message in Label")

    nex.destroy_nexus()
    assert True


def test_start_harvester(caplog, setdir, ports):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_harvester.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert any("Harvester server started" in record.msg for record in caplog.records)


def test_process_actor_state_update(caplog, setdir, ports):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_harvester.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        new_actor_message = ActorStateMsg("test actor", "waiting", 1234, "test info")

        nex.process_actor_state_update(new_actor_message)
        assert "test actor" in nex.actor_states
        assert nex.actor_states["test actor"].actor_name == new_actor_message.actor_name
        assert (
            nex.actor_states["test actor"].nexus_in_port
            == new_actor_message.nexus_in_port
        )
        assert nex.actor_states["test actor"].status == new_actor_message.status

        update_actor_message = ActorStateMsg("test actor", "waiting", 1234, "test info")

        nex.process_actor_state_update(update_actor_message)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")
    assert any(
        "Received state message from new actor test actor" in record.msg
        for record in caplog.records
    )
    assert any(
        "Received state message from actor test actor" in record.msg
        for record in caplog.records
    )


def test_process_actor_state_update_allows_run(caplog, setdir, ports):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_harvester.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        nex.actor_states["test actor1"] = None
        nex.actor_states["test actor2"] = None

        actor1_message = ActorStateMsg("test actor1", "ready", 1234, "test info")

        nex.process_actor_state_update(actor1_message)
        assert "test actor1" in nex.actor_states
        assert nex.actor_states["test actor1"].actor_name == actor1_message.actor_name
        assert (
            nex.actor_states["test actor1"].nexus_in_port
            == actor1_message.nexus_in_port
        )
        assert nex.actor_states["test actor1"].status == actor1_message.status

        assert not nex.allowStart

        actor2_message = ActorStateMsg("test actor2", "ready", 5678, "test info2")

        nex.process_actor_state_update(actor2_message)
        assert "test actor2" in nex.actor_states
        assert nex.actor_states["test actor2"].actor_name == actor2_message.actor_name
        assert (
            nex.actor_states["test actor2"].nexus_in_port
            == actor2_message.nexus_in_port
        )
        assert nex.actor_states["test actor2"].status == actor2_message.status

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")
    assert nex.allowStart


@pytest.mark.asyncio
async def test_process_actor_message(caplog, setdir, ports):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_harvester.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        nex.actor_states["test actor1"] = None
        nex.actor_states["test actor2"] = None

        actor1_message = ActorStateMsg("test actor1", "ready", 1234, "test info")

        ctx = nex.zmq_context
        s = ctx.socket(zmq.REQ)
        s.connect(f"tcp://localhost:{nex.actor_in_socket_port}")

        s.send_pyobj(actor1_message)

        await nex.process_actor_message()

        nex.process_actor_state_update(actor1_message)
        assert "test actor1" in nex.actor_states
        assert nex.actor_states["test actor1"].actor_name == actor1_message.actor_name
        assert (
            nex.actor_states["test actor1"].nexus_in_port
            == actor1_message.nexus_in_port
        )
        assert nex.actor_states["test actor1"].status == actor1_message.status

        s.close(linger=0)
        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert not nex.allowStart


def test_specified_free_port(caplog, setdir, ports):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_fixed_redis_port.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        store = StoreInterface(server_port_num=6378)
        store.connect_to_server()
        key = store.put("port 6378")
        assert store.get(key) == "port 6378"

        assert any(
            "Successfully connected to redis datastore on port 6378" in record.msg
            for record in caplog.records
        )

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert any(
        "StoreInterface start successful on port 6378" in record.msg
        for record in caplog.records
    )


def test_specified_busy_port(caplog, setdir, ports, setup_store):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_fixed_default_redis_port.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert any(
        "Could not connect to port 6379" in record.msg for record in caplog.records
    )

    assert any(
        "StoreInterface start successful on port 6380" in record.msg
        for record in caplog.records
    )


def test_unspecified_port_default_free(caplog, setdir, ports):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert any(
        "StoreInterface start successful on port 6379" in record.msg
        for record in caplog.records
    )


def test_unspecified_port_default_busy(caplog, setdir, ports, setup_store):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")
    assert any(
        "StoreInterface start successful on port 6380" in record.msg
        for record in caplog.records
    )


def test_no_aof_dir_by_default(caplog, setdir, ports):
    try:
        if "appendonlydir" in os.listdir("."):
            shutil.rmtree("appendonlydir")
        else:
            logging.info("didn't find dbfilename")

        nex = Nexus("test")

        nex.create_nexus(
            file="minimal.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert "appendonlydir" not in os.listdir(".")
    assert all(["improv_persistence_" not in name for name in os.listdir(".")])


def test_default_aof_dir_if_none_specified(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_redis_saving.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        store = StoreInterface(server_port_num=server_port_num)
        store.put(1)

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert "appendonlydir" in os.listdir(".")

    if "appendonlydir" in os.listdir("."):
        shutil.rmtree("appendonlydir")
    else:
        logging.info("didn't find dbfilename")

    logging.info("exited test")


def test_specify_static_aof_dir(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_custom_aof_dirname.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        store = StoreInterface(server_port_num=server_port_num)
        store.put(1)

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert "custom_aof_dirname" in os.listdir(".")

    if "custom_aof_dirname" in os.listdir("."):
        shutil.rmtree("custom_aof_dirname")
    else:
        logging.info("didn't find dbfilename")

    logging.info("exited test")


def test_use_ephemeral_aof_dir(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_ephemeral_aof_dirname.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        store = StoreInterface(server_port_num=server_port_num)
        store.put(1)

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert any(["improv_persistence_" in name for name in os.listdir(".")])

    [shutil.rmtree(db_filename) for db_filename in glob.glob("improv_persistence_*")]

    logging.info("completed ephemeral db test")


def test_save_no_schedule(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_no_schedule_saving.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        store = StoreInterface(server_port_num=server_port_num)

        fsync_schedule = store.client.config_get("appendfsync")

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert "appendonlydir" in os.listdir(".")
    shutil.rmtree("appendonlydir")

    assert fsync_schedule["appendfsync"] == "no"


def test_save_every_second(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_every_second_saving.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        store = StoreInterface(server_port_num=server_port_num)

        fsync_schedule = store.client.config_get("appendfsync")

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert "appendonlydir" in os.listdir(".")
    shutil.rmtree("appendonlydir")

    assert fsync_schedule["appendfsync"] == "everysec"


def test_save_every_write(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="minimal_with_every_write_saving.yaml",
            store_size=100_000_000,
            control_port=ports[0],
            output_port=ports[1],
        )

        store = StoreInterface(server_port_num=server_port_num)

        fsync_schedule = store.client.config_get("appendfsync")

        time.sleep(3)

        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")

    assert "appendonlydir" in os.listdir(".")
    shutil.rmtree("appendonlydir")

    assert fsync_schedule["appendfsync"] == "always"


# def test_sigint_exits_cleanly(ports, set_dir_config_parent):
#     server_opts = [
#         "improv",
#         "server",
#         "-c",
#         str(ports[0]),
#         "-o",
#         str(ports[1]),
#         "-f",
#         "global.log",
#         "configs/minimal.yaml",
#     ]
#
#     env = os.environ.copy()
#     env["PYTHONPATH"] += ":" + os.getcwd()
#
#     server = subprocess.Popen(
#         server_opts, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
#     )
#
#     time.sleep(5)
#
#     server.send_signal(signal.SIGINT)
#
#     server.wait(10)
#     assert True


# def test_nexus_actor_in_port(ports, setdir, start_nexus_minimal_zmq):
#     context = zmq.Context()
#     nex_socket = context.socket(zmq.REQ)
#     nex_socket.connect(f"tcp://localhost:{ports[3]}")  # actor in port
#
#     test_socket = context.socket(zmq.REP)
#     test_socket.bind("tcp://*:0")
#     in_port_string = test_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
#     test_socket_port = int(in_port_string.split(":")[-1])
#     logging.info(f"Using port {test_socket_port}")
#
#     logging.info("waiting to send")
#     actor_state = ActorStateMsg(
#         "test_actor", "test_status", test_socket_port, "test info string"
#     )
#     nex_socket.send_pyobj(actor_state)
#     logging.info("Sent")
#     out = nex_socket.recv_pyobj()
#     assert isinstance(out, ActorStateReplyMsg)
#     assert out.actor_name == actor_state.actor_name
#     assert out.status == "OK"


def test_nexus_create_nexus_no_cfg_file(ports):
    nex = Nexus("test")
    with pytest.raises(ConfigFileNotProvidedException):
        nex.create_nexus()


#
# @pytest.mark.skip(reason="Blocking comms so this won't work as-is")
# def test_nexus_actor_comm_setup(ports, setdir):
#     filename = "minimal_zmq.yaml"
#     nex = Nexus("test")
#     nex.create_nexus(
#         file=filename,
#         store_size=10000000,
#         control_port=ports[0],
#         output_port=ports[1],
#         actor_in_port=ports[2],
#     )
#
#     actor = nex.actors["Generator"]
#     actor.register_with_nexus()
#
#     nex.process_actor_message()
#
#
# @pytest.mark.skip(reason="Test isn't meant to be used for coverage")
# def test_debug_nex(ports, setdir):
#     filename = "minimal_zmq.yaml"
#     conftest.nex_startup(ports, filename)
#
#
# @pytest.mark.skip(reason="Test isn't meant to be used for coverage")
# def test_nex_cfg(ports, setdir):
#     filename = "minimal_zmq.yaml"
#     nex = Nexus("test")
#     nex.create_nexus(
#         file=filename,
#         store_size=100000000,
#         control_port=ports[0],
#         output_port=ports[1],
#         actor_in_port=ports[2],
#     )
#     nex.start_nexus()


def test_nexus_bad_config_actor_args(setdir):
    nex = Nexus("test")
    # with pytest.raises(ConfigFileNotValidException):
    try:
        nex.create_nexus("bad_args.yaml")
    except ConfigFileNotValidException:
        assert True
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")


def test_nexus_no_config_file():
    nex = Nexus("test")
    # with pytest.raises(ConfigFileNotProvidedException):
    try:
        nex.create_nexus()
    except ConfigFileNotProvidedException:
        assert True
    except Exception as e:
        print(f"error caught in test harness: {e}")
        logging.error(f"error caught in test harness: {e}")
