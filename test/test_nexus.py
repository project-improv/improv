import glob
import shutil
import time
import os
import pytest
import logging
import subprocess
import signal
import yaml

from improv.nexus import Nexus
from improv.store import StoreInterface

# from improv.actor import Actor
# from improv.store import StoreInterface

SERVER_COUNTER = 0


@pytest.fixture
def ports():
    global SERVER_COUNTER
    CONTROL_PORT = 5555
    OUTPUT_PORT = 5556
    LOGGING_PORT = 5557
    yield (
        CONTROL_PORT + SERVER_COUNTER,
        OUTPUT_PORT + SERVER_COUNTER,
        LOGGING_PORT + SERVER_COUNTER,
    )
    SERVER_COUNTER += 3


@pytest.fixture
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__) + "/configs")
    yield None
    os.chdir(prev)


@pytest.fixture
def sample_nex(setdir, ports):
    nex = Nexus("test")
    nex.createNexus(
        file="good_config.yaml",
        store_size=40000000,
        control_port=ports[0],
        output_port=ports[1],
    )
    yield nex
    nex.destroyNexus()


# @pytest.fixture
# def setup_store(setdir):
#     """ Fixture to set up the store subprocess with 10 mb.

#     This fixture runs a subprocess that instantiates the store with a
#     memory of 10 megabytes. It specifies that "/tmp/store/" is the
#     location of the store socket.

#     Yields:
#         StoreInterface: An instance of the store.

#     TODO:
#         Figure out the scope.
#     """
#     setdir
#     p = subprocess.Popen(
#         ['plasma_store', '-s', '/tmp/store/', '-m', str(10000000)],\
#         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     store = StoreInterface(store_loc = "/tmp/store/")
#     yield store
#     p.kill()


def test_init(setdir):
    # store = setup_store
    nex = Nexus("test")
    assert str(nex) == "test"


@pytest.mark.parametrize(
    "cfg_name",
    [
        "good_config.yaml",
        "good_config_plasma.yaml",
    ],
)
def test_createNexus(setdir, ports, cfg_name):
    nex = Nexus("test")
    nex.createNexus(file=cfg_name, control_port=ports[0], output_port=ports[1])
    assert list(nex.comm_queues.keys()) == [
        "GUI_comm",
        "Acquirer_comm",
        "Analysis_comm",
    ]
    assert list(nex.sig_queues.keys()) == ["Acquirer_sig", "Analysis_sig"]
    assert list(nex.data_queues.keys()) == ["Acquirer.q_out", "Analysis.q_in"]
    assert list(nex.actors.keys()) == ["Acquirer", "Analysis"]
    assert list(nex.flags.keys()) == ["quit", "run", "load"]
    assert nex.processes == []
    nex.destroyNexus()
    assert True


def test_config_logged(setdir, ports, caplog):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_settings.yaml", control_port=ports[0], output_port=ports[1]
    )
    nex.destroyNexus()
    assert any(
        [
            "not_relevant: for testing purposes" in record.msg
            for record in caplog.records
        ]
    )


def test_loadConfig(sample_nex):
    nex = sample_nex
    nex.loadConfig("good_config.yaml")
    assert set(nex.comm_queues.keys()) == set(
        ["Acquirer_comm", "Analysis_comm", "GUI_comm"]
    )


def test_argument_config_precedence(setdir, ports):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_settings.yaml",
        control_port=ports[0],
        output_port=ports[1],
        store_size=11_000_000,
        use_watcher=True,
    )
    cfg = nex.config.settings
    nex.destroyNexus()
    assert cfg["control_port"] == ports[0]
    assert cfg["output_port"] == ports[1]
    assert cfg["store_size"] == 20_000_000
    assert not cfg["use_watcher"]


def test_settings_override_random_ports(setdir, ports):
    config_file = "minimal_with_settings.yaml"
    nex = Nexus("test")
    with open(config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)["settings"]
    control_port, output_port = nex.createNexus(
        file=config_file, control_port=0, output_port=0
    )
    nex.destroyNexus()
    assert control_port == cfg["control_port"]
    assert output_port == cfg["output_port"]


# delete this comment later
@pytest.mark.skip(reason="unfinished")
def test_startNexus(sample_nex):
    nex = sample_nex
    nex.startNexus()
    assert [p.name for p in nex.processes] == ["Acquirer", "Analysis"]
    nex.destroyNexus()


# @pytest.mark.skip(reason="This test is unfinished")
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
    nex.createNexus(file=cfg_name, control_port=ports[0], output_port=ports[1])
    logging.info(cfg_name)

    # Check for actors

    act_lst = list(nex.actors)
    lnk_lst = list(nex.sig_queues)

    nex.destroyNexus()

    assert actor_list == act_lst
    assert link_list == lnk_lst
    act_lst = []
    lnk_lst = []
    assert True


@pytest.mark.parametrize(
    "cfg_name",
    [
        "single_actor.yaml",
        "single_actor_plasma.yaml",
    ],
)
def test_single_actor(setdir, ports, cfg_name):
    nex = Nexus("test")
    with pytest.raises(AttributeError):
        nex.createNexus(
            file="single_actor.yaml", control_port=ports[0], output_port=ports[1]
        )

    nex.destroyNexus()


def test_cyclic_graph(setdir, ports):
    nex = Nexus("test")
    nex.createNexus(
        file="cyclic_config.yaml", control_port=ports[0], output_port=ports[1]
    )
    assert True
    nex.destroyNexus()


def test_blank_cfg(setdir, caplog, ports):
    nex = Nexus("test")
    with pytest.raises(TypeError):
        nex.createNexus(
            file="blank_file.yaml", control_port=ports[0], output_port=ports[1]
        )
    assert any(
        ["The config file is empty" in record.msg for record in list(caplog.records)]
    )
    nex.destroyNexus()


# def test_hasGUI_True(setdir):
#     setdir
#     nex = Nexus("test")
#     nex.createNexus(file="basic_demo_with_GUI.yaml")

#     assert True
#     nex.destroyNexus()

# @pytest.mark.skip(reason="This test is unfinished.")
# def test_hasGUI_False():
#     assert True


@pytest.mark.skip(reason="unfinished")
def test_queue_message(setdir, sample_nex):
    nex = sample_nex
    nex.startNexus()
    time.sleep(20)
    nex.setup()
    time.sleep(20)
    nex.run()
    time.sleep(10)
    acq_comm = nex.comm_queues["Acquirer_comm"]
    acq_comm.put("Test Message")

    assert nex.comm_queues is None
    nex.destroyNexus()
    assert True


@pytest.mark.asyncio
@pytest.mark.skip(reason="This test is unfinished.")
async def test_queue_readin(sample_nex, caplog):
    nex = sample_nex
    nex.startNexus()
    # cqs = nex.comm_queues
    # assert cqs == None
    assert [record.msg for record in caplog.records] is None
    # cqs["Acquirer_comm"].put('quit')
    # assert "quit" == cqs["Acquirer_comm"].get()
    # await nex.pollQueues()
    assert True


@pytest.mark.skip(reason="This test is unfinished.")
def test_queue_sendout():
    assert True


@pytest.mark.skip(reason="This test is unfinished.")
def test_run_sig():
    assert True


@pytest.mark.skip(reason="This test is unfinished.")
def test_setup_sig():
    assert True


@pytest.mark.skip(reason="This test is unfinished.")
def test_quit_sig():
    assert True


@pytest.mark.skip(reason="This test is unfinished.")
def test_usehdd_True():
    assert True


@pytest.mark.skip(reason="This test is unfinished.")
def test_usehdd_False():
    assert True


def test_startstore(caplog):
    nex = Nexus("test")
    nex._startStoreInterface(10000000)  # 10 MB store

    assert any(
        "StoreInterface start successful" in record.msg for record in caplog.records
    )

    nex._closeStoreInterface()
    nex.destroyNexus()
    assert True


def test_closestore(caplog):
    nex = Nexus("test")
    nex._startStoreInterface(10000)
    nex._closeStoreInterface()

    assert any(
        "StoreInterface close successful" in record.msg for record in caplog.records
    )

    # write to store

    with pytest.raises(AttributeError):
        nex.p_StoreInterface.put("Message in", "Message in Label")

    nex.destroyNexus()
    assert True


def test_specified_free_port(caplog, setdir, ports):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_fixed_redis_port.yaml",
        store_size=10000000,
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

    nex.destroyNexus()

    assert any(
        "StoreInterface start successful on port 6378" in record.msg
        for record in caplog.records
    )


def test_specified_busy_port(caplog, setdir, ports, setup_store):
    nex = Nexus("test")
    with pytest.raises(Exception, match="Could not start Redis on specified port."):
        nex.createNexus(
            file="minimal_with_fixed_default_redis_port.yaml",
            store_size=10000000,
            control_port=ports[0],
            output_port=ports[1],
        )

    nex.destroyNexus()

    assert any(
        "Could not start Redis on specified port number." in record.msg
        for record in caplog.records
    )


def test_unspecified_port_default_free(caplog, setdir, ports):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    nex.destroyNexus()

    assert any(
        "StoreInterface start successful on port 6379" in record.msg
        for record in caplog.records
    )


def test_unspecified_port_default_busy(caplog, setdir, ports, setup_store):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    nex.destroyNexus()
    assert any(
        "StoreInterface start successful on port 6380" in record.msg
        for record in caplog.records
    )


def test_no_aof_dir_by_default(caplog, setdir, ports):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    nex.destroyNexus()

    assert "appendonlydir" not in os.listdir(".")
    assert all(["improv_persistence_" not in name for name in os.listdir(".")])


def test_default_aof_dir_if_none_specified(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_redis_saving.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    store = StoreInterface(server_port_num=server_port_num)
    store.put(1)

    time.sleep(3)

    nex.destroyNexus()

    assert "appendonlydir" in os.listdir(".")

    if "appendonlydir" in os.listdir("."):
        shutil.rmtree("appendonlydir")
    else:
        logging.info("didn't find dbfilename")

    logging.info("exited test")


def test_specify_static_aof_dir(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_custom_aof_dirname.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    store = StoreInterface(server_port_num=server_port_num)
    store.put(1)

    time.sleep(3)

    nex.destroyNexus()

    assert "custom_aof_dirname" in os.listdir(".")

    if "custom_aof_dirname" in os.listdir("."):
        shutil.rmtree("custom_aof_dirname")
    else:
        logging.info("didn't find dbfilename")

    logging.info("exited test")


def test_use_ephemeral_aof_dir(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_ephemeral_aof_dirname.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    store = StoreInterface(server_port_num=server_port_num)
    store.put(1)

    time.sleep(3)

    nex.destroyNexus()

    assert any(["improv_persistence_" in name for name in os.listdir(".")])

    [shutil.rmtree(db_filename) for db_filename in glob.glob("improv_persistence_*")]

    logging.info("completed ephemeral db test")


def test_save_no_schedule(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_no_schedule_saving.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    store = StoreInterface(server_port_num=server_port_num)

    fsync_schedule = store.client.config_get("appendfsync")

    nex.destroyNexus()

    assert "appendonlydir" in os.listdir(".")
    shutil.rmtree("appendonlydir")

    assert fsync_schedule["appendfsync"] == "no"


def test_save_every_second(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_every_second_saving.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    store = StoreInterface(server_port_num=server_port_num)

    fsync_schedule = store.client.config_get("appendfsync")

    nex.destroyNexus()

    assert "appendonlydir" in os.listdir(".")
    shutil.rmtree("appendonlydir")

    assert fsync_schedule["appendfsync"] == "everysec"


def test_save_every_write(caplog, setdir, ports, server_port_num):
    nex = Nexus("test")
    nex.createNexus(
        file="minimal_with_every_write_saving.yaml",
        store_size=10000000,
        control_port=ports[0],
        output_port=ports[1],
    )

    store = StoreInterface(server_port_num=server_port_num)

    fsync_schedule = store.client.config_get("appendfsync")

    nex.destroyNexus()

    assert "appendonlydir" in os.listdir(".")
    shutil.rmtree("appendonlydir")

    assert fsync_schedule["appendfsync"] == "always"


@pytest.mark.skip(reason="Nexus no longer deletes files on shutdown. Nothing to test.")
def test_store_already_deleted_issues_warning(caplog):
    nex = Nexus("test")
    nex._startStoreInterface(10000)
    store_location = nex.store_loc
    StoreInterface(store_loc=nex.store_loc)
    os.remove(nex.store_loc)
    nex.destroyNexus()
    assert any(
        "StoreInterface file {} is already deleted".format(store_location) in record.msg
        for record in caplog.records
    )


@pytest.mark.skip(reason="unfinished")
def test_actor_sub(setdir, capsys, monkeypatch, ports):
    monkeypatch.setattr("improv.nexus.input", lambda: "setup\n")
    cfg_file = "sample_config.yaml"
    nex = Nexus("test")

    nex.createNexus(
        file=cfg_file, store_size=4000, control_port=ports[0], output_port=ports[1]
    )
    print("Nexus Created")

    nex.startNexus()
    print("Nexus Started")
    # time.sleep(5)
    # print("Printing...")
    # subprocess.Popen(["setup"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # time.sleep(2)
    # subprocess.Popen(["run"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # time.sleep(5)
    # subprocess.Popen(["quit"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    nex.destroyNexus()
    assert True


@pytest.mark.skip(
    reason="skipping to prevent issues with orphaned stores. TODO fix this"
)
def test_sigint_exits_cleanly(ports, tmp_path):
    server_opts = [
        "improv",
        "server",
        "-c",
        str(ports[0]),
        "-o",
        str(ports[1]),
        "-f",
        tmp_path / "global.log",
    ]

    server = subprocess.Popen(
        server_opts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    server.send_signal(signal.SIGINT)

    server.wait(10)
    assert True
