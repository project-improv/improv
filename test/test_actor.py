import os
import psutil
import pytest
import zmq

from improv.actor import AbstractActor as Actor
from improv.messaging import ActorStateMsg, ActorStateReplyMsg
from improv.store import StoreInterface
from improv.link import ZmqLink

# set global_variables

pytest.example_string_links = {}
pytest.example_links = {}


@pytest.fixture
def init_actor():
    """Fixture to initialize and teardown an instance of actor."""

    act = Actor("Test")
    yield act
    act = None


@pytest.fixture
def example_string_links():
    """Fixture to provide a commonly used test input."""

    pytest.example_string_links = {"1": "one", "2": "two", "3": "three"}
    return pytest.example_string_links


@pytest.fixture
def example_links(setup_store, server_port_num):
    """Fixture to provide link objects as test input and setup store."""
    StoreInterface(server_port_num=server_port_num)

    ctx = zmq.Context()
    s = ctx.socket(zmq.PUSH)

    links = [ZmqLink(s, "test") for i in range(2)]
    link_dict = {links[i].name: links[i] for i, l in enumerate(links)}
    pytest.example_links = link_dict
    yield pytest.example_links
    s.close(linger=0)
    ctx.destroy(linger=0)


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("q_watchout", None),
        ("name", "Test"),
        ("links", {}),
        ("lower_priority", False),
        ("q_in", None),
        ("q_out", None),
    ],
)
def test_default_init(attribute, expected, init_actor):
    """Tests if the default init attributes are as expected."""

    atr = getattr(init_actor, attribute)

    assert atr == expected


def test_repr_default_initialization(init_actor):
    """Test if the actor representation has the right dict keys."""

    act = init_actor
    rep = act.__repr__()
    assert rep == "Test: dict_keys([])"


def test_repr(example_string_links):
    """Test if the actor representation has the right, nonempty, dict."""

    act = Actor("Test")
    act.set_links(example_string_links)
    assert act.__repr__() == "Test: dict_keys(['1', '2', '3'])"


def test_set_store_interface(setup_store, server_port_num):
    """Tests if the store is started and linked with the actor."""

    act = Actor("Acquirer", server_port_num)
    store = StoreInterface(server_port_num=server_port_num)
    act.set_store_interface(store.client)
    assert act.client is store.client


@pytest.mark.parametrize(
    "links", [pytest.example_string_links, ({}), pytest.example_links, None]
)
def test_set_links(links):
    """Tests if the actors links can be set to certain values."""

    act = Actor("test")
    act.set_links(links)
    assert act.links == links


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_set_link_in(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the input queue."""

    act = init_actor
    act.set_links(links)
    if links is not None:
        act.set_link_in("input_q")
        expected.update({"q_in": "input_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.set_link_in("input_queue")


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_set_link_out(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the output queue."""

    act = init_actor
    act.set_links(links)
    if links is not None:
        act.set_link_out("output_q")
        expected.update({"q_out": "output_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.set_link_in("output_queue")


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_set_link_watch(
    init_actor, example_string_links, example_links, links, expected
):
    """Tests if we can set the watch queue."""

    act = init_actor
    act.set_links(links)
    if links is not None:
        act.set_link_watch("watch_q")
        expected.update({"q_watchout": "watch_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.set_link_in("input_queue")


def test_add_link(setup_store):
    """Tests if a link can be added to the dictionary of links."""

    act = Actor("test")
    links = {"1": "one", "2": "two"}
    act.set_links(links)
    newName = "3"
    newLink = "three"
    act.add_link(newName, newLink)
    links.update({"3": "three"})

    # trying to check for two separate conditions while being able to
    # distinguish between them should an error be raised
    passes = []
    err_messages = []

    if act.get_links()["3"] == "three":
        passes.append(True)
    else:
        passes.append(False)
        err_messages.append(
            "Error:\
            actor.getLinks()['3'] is not equal to \"three\""
        )

    if act.get_links() == links:
        passes.append(True)
    else:
        passes.append("False")
        err_messages.append(
            "Error:\
            actor.getLinks() is not equal to the links dictionary"
        )

    err_out = "\n".join(err_messages)
    assert all(passes), f"The following errors occurred: {err_out}"


def test_get_links(init_actor, example_string_links):
    """Tests if we can access the dictionary of links.

    TODO:
        Add more parametrized test cases.
    """

    act = init_actor
    links = example_string_links
    act.set_links(links)

    assert act.get_links() == {"1": "one", "2": "two", "3": "three"}


def test_run(init_actor):
    """Tests if actor.run raises an error."""
    act = init_actor
    with pytest.raises(NotImplementedError):
        act.run()


def test_change_priority(init_actor):
    """Tests if we are able to change the priority of an actor."""

    act = init_actor
    act.lower_priority = True
    act.change_priority()

    assert psutil.Process(os.getpid()).nice() == 19


def test_actor_connection(setup_store, server_port_num):
    """Test if the links between actors are established correctly.

    This test instantiates two actors with different names, then instantiates
    a Link object linking the two actors. A string is put to the input queue of
    one actor. Then, in the other actor, it is removed from the queue, and
    checked to verify it matches the original message.
    """
    assert True


def test_actor_registration_with_nexus(ports, zmq_actor):
    context = zmq.Context()
    nex_socket = context.socket(zmq.REP)
    nex_socket.bind(f"tcp://*:{ports[3]}")  # actor in port

    zmq_actor.start()

    res = nex_socket.recv_pyobj()
    assert isinstance(res, ActorStateMsg)

    nex_socket.send_pyobj(ActorStateReplyMsg("test", "OK", ""))

    zmq_actor.terminate()
    zmq_actor.join(10)


# TODO: register with broker test
