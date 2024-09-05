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
        store_size=4000,
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


def test_createNexus(setdir, ports):
    nex = Nexus("test")
    nex.createNexus(
        file="good_config.yaml", control_port=ports[0], output_port=ports[1]
    )
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
        use_hdd=True,
        use_watcher=True,
    )
    cfg = nex.config.settings
    nex.destroyNexus()
    assert cfg["control_port"] == ports[0]
    assert cfg["output_port"] == ports[1]
    assert cfg["store_size"] == 20_000_000
    assert not cfg["use_hdd"]
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


def test_single_actor(setdir, ports):
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


def test_startstore(caplog, set_store_loc):
    nex = Nexus("test")
    nex._startStoreInterface(10000)  # 10 kb store

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
