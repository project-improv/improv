import os
import psutil
import pytest
import subprocess
from improv.link import Link, AsyncQueue
from improv.actor import Actor as actor
from improv.store import Limbo as limbo

#set global_variables

pytest.example_string_links =  {}
pytest.example_links = {}

@pytest.fixture
def setup_store(scope="module"):
    """ Fixture to set up the store subprocess with 10 mb.
    """

    p = subprocess.Popen(
        ["plasma_store", "-s", "/tmp/store", "-m", str(10000000)],\
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

    pytest.example_string_links = {"1": "one", "2": "two", "3": "three"}
    yield pytest.example_string_links
    pytest.example_string_links = {"1": "one", "2": "two", "3": "three"}


@pytest.fixture
def example_links(setup_store):
    """ Fixture to provide link objects as test input and setup store.
    """
    setup_store
    lmb = limbo(store_loc="/tmp/store")

    acts = [actor("act" + str(i)) for i in range(1, 5)] #range must be even

    links = [Link("L" + str(i + 1), acts[i], acts[i + 1], lmb) for i in range(len(acts) // 2)]
    link_dict = {links[i].name: links[i] for i, l in enumerate(links)}
    pytest.example_links = link_dict
    yield pytest.example_links
    pytest.example_links = link_dict


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


def test_repr(example_string_links):
    """ Test if the actor representation has the right, nonempty, dict.
    """

    act = actor("Test")
    act.setLinks(example_string_links)
    assert act.__repr__() == "Test: dict_keys([\'1\', \'2\', \'3\'])"


def test_setStore(setup_store):
    """ Tests if the store is started and linked with the actor.
    """

    act = actor("Acquirer")
    lmb = limbo(store_loc="/tmp/store")
    act.setStore(lmb.client)
    assert act.client is lmb.client


@pytest.mark.parametrize("links", [
    (pytest.example_string_links),
    ({}), 
    (pytest.example_links),
    (None)
])
def test_setLinks(links):
    """ Tests if the actors links can be set to certain values.
    """

    act = actor("test")
    act.setLinks(links)
    assert act.links == links 


@pytest.mark.parametrize("qc, qs", [
    ("comm", "sig"),
    (None, None),
    ("", ""),
    ("LINK", "LINK") #these are placeholder names (store is not setup)
])
def test_setCommLinks(example_links, qc, qs, init_actor, setup_store):
    """ Tests if commLinks can be added to the actor"s links.
    """

    if (qc == "LINK" and qs == "LINK"):
        qc = Link("L1", actor("1"), actor("2"), limbo(store_loc = "/tmp/store"))
        qs = Link("L2", actor("3"), actor("4"), limbo(store_loc = "/tmp/store"))
    act = init_actor
    act.setLinks(example_links)
    act.setCommLinks(qc, qs)
    
    example_links.update({"q_comm": qc, "q_sig": qs}) 
    assert act.links == example_links


@pytest.mark.parametrize("links, expected", [
    (pytest.example_string_links, pytest.example_string_links),
    (pytest.example_links, pytest.example_links),
    ({}, {}),
    (None, TypeError)
])
def test_setLinkIn(init_actor, example_string_links, example_links, links, expected):
    """ Tests if we can set the input queue.
    """

    act = init_actor
    act.setLinks(links)
    if (links != None):
        act.setLinkIn("input_q")
        expected.update({"q_in": "input_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("input_queue")


@pytest.mark.parametrize("links, expected", [
    (pytest.example_string_links, pytest.example_string_links),
    (pytest.example_links, pytest.example_links),
    ({}, {}),
    (None, TypeError)
])
def test_setLinkOut(init_actor, example_string_links, example_links, links, expected):
    """ Tests if we can set the output queue.
    """

    act = init_actor
    act.setLinks(links)
    if (links != None):
        act.setLinkOut("output_q")
        expected.update({"q_out": "output_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("output_queue")


@pytest.mark.parametrize("links, expected", [
    (pytest.example_string_links, pytest.example_string_links),
    (pytest.example_links, pytest.example_links),
    ({}, {}),
    (None, TypeError)
])
def test_setLinkWatch(init_actor, example_string_links, example_links, links, expected):
    """ Tests if we can set the watch queue.
    """

    act = init_actor
    act.setLinks(links)
    if (links != None):
        act.setLinkWatch("watch_q")
        expected.update({"q_watchout": "watch_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("input_queue")


def test_addLink(setup_store):
    """ Tests if a link can be added to the dictionary of links.
    """

    act = actor("test")
    links = {"1": "one", "2": "two"}
    act.setLinks(links)
    newName = "3"
    newLink = "three"
    act.addLink(newName, newLink)
    links.update({"3": "three"})

    #trying to check for two separate conditions while being able to
    #distinguish between them should an error be raised
    passes = []
    err_messages = []

    if (act.getLinks()["3"], "three"):
        passes.append(True)
    else:
        passes.append(False)
        err_messages.append("Error:\
            actor.getLinks()['3'] is not equal to \"three\"")

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

    assert act.getLinks() == {"1": "one", "2": "two", "3": "three"}


def test_setup_unimplemented(init_actor, example_string_links):
    """ Checks if actor.setup raises an error.
    """ 

    with pytest.raises(NotImplementedError):
        act = init_actor
        act.setup()


@pytest.mark.skip(reason="this is something we\'ll do later because\
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
    """ Tests if actor.run raises an error.
    """ 

    with pytest.raises(NotImplementedError):
        act = init_actor
        act.run()


def test_changePriority(init_actor):
    """ Tests if we are able to change the priority of an actor.
    """

    act = init_actor
    act.lower_priority = True
    act.changePriority()

    assert psutil.Process(os.getpid()).nice() == 19


def test_actor_connection(setup_store):
    """ Test if the links between actors are established correctly.

    This test instantiates two actors with different names, then instantiates
    a Link object linking the two actors. A string is put to the input queue of 
    one actor. Then, in the other actor, it is removed from the queue, and 
    checked to verify it matches the original message.
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
