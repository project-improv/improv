import os
import psutil
import pytest
import subprocess
from improv.link import Link, AsyncQueue
from improv.actor import Actor as actor
from improv.store import Limbo as limbo

#set global_variables

pytest.example_string_links = {'1': "one", '2': "two", '3': "three"}

@pytest.fixture
def setup_store():
    """ Fixture to set up the store subprocess.
    """

    p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)],\
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    yield p
    p.kill()

@pytest.fixture
def init_actor():
    """ Fixture to initialize and teardown an instance of actor.
    """

    act = actor("Test")
    yield act
    act = None

@pytest.fixture
def example_string_links():
    """ Fixture to provide a commonly used test input.
    """

    pytest.example_string_links = {'1': "one", '2': "two", '3': "three"}
    yield pytest.example_string_links
    pytest.example_string_links = {'1': "one", '2': "two", '3': "three"}

@pytest.mark.parametrize("attribute, expected", [
    ("q_watchout", None),
    ("name", "Test"),
    ("links", {}),
    ("done", False),
    ("lower_priority", False),
    ("q_in", None),
    ("q_out", None)
])
def test_default_init(attribute, expected, init_actor):
    """ Tests if the default init attributes are as expected.
    """

    atr = getattr(init_actor, attribute)

    assert atr == expected

def test_repr_default_initialization(init_actor):
    """ Test if the actor representation has the right dict keys.
    """

    act = init_actor
    rep = act.__repr__()
    assert rep == "Test: dict_keys([])"

def test_setStore(setup_store):
    """ Tests if the store is started and linked with the actor.
    """

    act = actor("Acquirer")
    lmb = limbo(store_loc="/tmp/store")
    act.setStore(lmb.client)
    assert act.client is lmb.client

@pytest.mark.parametrize("links, expected", [
    (pytest.example_string_links, pytest.example_string_links),
    ({}, {}),
    (None, None)
])
def test_setLinks(links, expected):
    """ Tests if the actors links can be set to certain values.
    """

    act = actor("test")
    act.setLinks(links)
    assert act.links == expected

@pytest.mark.parametrize("links, qc, qs, expected", [
    (pytest.example_string_links, "comm", "sig", {
    '1': "one", '2': "two", '3': "three", "q_comm": "comm", "q_sig": "sig"}),
    (pytest.example_string_links, None, None, {'1': "one", '2': "two", '3': "three",
    "q_comm": None, "q_sig": None})
])
def test_setCommLinks(links, qc, qs, expected):
    """ Tests if commLinks can be added to the actor's links.
    """

    act = actor("Test")
    act.setLinks(links)
    act.setCommLinks(qc, qs)
    assert act.links == expected

def test_setLinkIn(init_actor, example_string_links):
    """ Tests if we can set the input queue.

    TODO:
        Add more parametrized test cases.
    """

    act = init_actor
    act.setLinks(example_string_links)
    act.setLinkIn("input_q")
    assert pytest.example_string_links == {
        '1': "one", '2': "two", '3': "three", "q_in": "input_q"}

def test_setLinkOut(init_actor, example_string_links):
    """ Tests if we can set the output queue.

    TODO:
        Add more parametrized test cases.
    """

    act = init_actor
    links = example_string_links
    act.setLinks(links)
    act.setLinkOut("output_q")
    assert act.links == {
        '1': "one", '2': "two", '3': "three", "q_out": "output_q"}

def test_setLinkWatch(init_actor, example_string_links):
    """ Tests if we can set the watch queue.

    TODO:
        Add more parametrized test cases.
    """

    act = init_actor
    link = example_string_links
    act.setLinks(link)
    act.setLinkWatch("watch_q")
    assert act.links == {
        '1': "one", '2': "two", '3': "three", "q_watchout": "watch_q"
    }

def test_addLink(setup_store):
    """ Tests if a link can be added to the dictionary of links.
    """

    act = actor("test")
    links = {'1': "one", '2': "two"}
    act.setLinks(links)
    newName = '3'
    newLink = "three"
    act.addLink(newName, newLink)
    links.update({'3': "three"})

    #trying to check for two separate conditions while being able to
    #distinguish between them should an error be raised
    passes = []
    err_messages = []

    if (act.getLinks()['3'], "three"):
        passes.append(True)
    else:
        passes.append(False)
        err_messages.append("Error:\
            actor.getLinks()['3']) is not equal to \"three\"")

    if (act.getLinks() == links):
        passes.append(True)
    else:
        passes.append("False")
        err_messages.append("Error:\
            actor.getLinks() is not equal to the links dictionary")

    err_out = "\n".join(err_messages)
    assert all(passes), f"The following errors occurred: {err_out}"

def test_getLinks(init_actor, example_string_links):
    """ Tests if we can access the dictionary of links.

    TODO:
        Add more parametrized test cases.
    """

    act = init_actor
    links = example_string_links
    act.setLinks(links)

    assert act.getLinks() == {'1': "one", '2': "two", '3': "three"}

def test_setup_unimplemented(init_actor, example_string_links):
    with pytest.raises(NotImplementedError):
        act = init_actor
        act.setup()

@pytest.mark.skip(reason="this is something we'll do later because\
                    we will subclass actor w/ watcher later")
def test_put(init_actor):
    """ Tests if data keys can be put to output links.

    TODO:
        Ask Anne to take a look.
    """

    act = init_actor
    act.put()
    assert True

def test_run(init_actor):
    with pytest.raises(NotImplementedError):
        act = init_actor
        act.run()

def test_changePriority(init_actor):
    """Tests if we are able to change the priority of an actor.
    """

    act = init_actor
    act.lower_priority = True
    act.changePriority()

    assert psutil.Process(os.getpid()).nice() == 19

def test_actor_connection(setup_store):
    """ Test if the links between actors are established correctly.

    TODO:
        Instantiate two actors
        Set links between them
        Check if a message can be sent between links
    """

    act1 = actor("a1")
    act2 = actor("a2")

    lmb = limbo(store_loc="/tmp/store")
    link = Link("L12", act1, act2, lmb)
    act1.setLinkIn(link)
    act2.setLinkOut(link)


    msg = "message"

    act1.q_in.put(msg)

    assert act2.q_out.get() == msg
