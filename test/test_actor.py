import os
import psutil
import pytest
import subprocess
import logging
from improv.link import Link  # , AsyncQueue
from improv.actor import AbstractActor as Actor
from improv.store import StoreInterface


# set global_variables

LOGGER = logging.getLogger(__name__)

pytest.example_string_links = {}
pytest.example_links = {}


@pytest.fixture()
def setup_store(set_store_loc, scope="module"):
    """Fixture to set up the store subprocess with 10 mb."""
    #print(f"set store loc: {set_store_loc}")
    LOGGER.info(f"set store loc: {set_store_loc}")
    p = subprocess.Popen(
        ["plasma_store", "-s", set_store_loc, "-m", str(10000000)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True
    )
    yield p

    print("about to wait: first time")
    print("about to kill")
    p.kill()
    print("about to wait")
    p.wait(10)


@pytest.fixture()
def init_actor(set_store_loc):
    """Fixture to initialize and teardown an instance of actor."""

    act = Actor("Test", set_store_loc)
    yield act
    act = None


@pytest.fixture()
def example_string_links():
    """Fixture to provide a commonly used test input."""

    pytest.example_string_links = {"1": "one", "2": "two", "3": "three"}
    return pytest.example_string_links


@pytest.fixture()
def example_links(setup_store, set_store_loc):
    """Fixture to provide link objects as test input and setup store."""
    StoreInterface(store_loc=set_store_loc)

    acts = [
        Actor("act" + str(i), set_store_loc) for i in range(1, 5)
    ]  # range must be even

    links = [
        Link("L" + str(i + 1), acts[i], acts[i + 1]) for i in range(len(acts) // 2)
    ]
    link_dict = {links[i].name: links[i] for i, l in enumerate(links)}
    pytest.example_links = link_dict
    return pytest.example_links


def test_CI_debug():
    """ For CI debug only; delete afterwards."""
    assert True

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


def test_repr_default_initialization(init_actor, set_store_loc):
    """Test if the actor representation has the right dict keys."""

    act = init_actor
    rep = act.__repr__()
    #print(f"\nstore_loc: {set_store_loc}\n")
    assert rep == "Test: dict_keys([])"


def test_repr(example_string_links, set_store_loc):
    """Test if the actor representation has the right, nonempty, dict."""

    act = Actor("Test", set_store_loc)
    act.setLinks(example_string_links)
    #print(f"set store loc: {set_store_loc}")
    assert act.__repr__() == "Test: dict_keys(['1', '2', '3'])"


def test_setStoreInterface(setup_store, set_store_loc):
    """Tests if the store is started and linked with the actor."""
    
    #print("HERE");
    LOGGER.info("here")
    act = Actor("Acquirer", set_store_loc)
    #print(f"store_loc: {set_store_loc}")
    LOGGER.info(f"store_loc: {set_store_loc}")
    store = StoreInterface(store_loc=set_store_loc)
    print("got store interface")
    print("about to connect actor")
    act.setStoreInterface(store.client)
    assert act.client is store.client
    print("all good!")

from pyarrow import plasma 
def test_foo(setup_store, set_store_loc):
    # act = Actor("Acquirer", set_store_loc)
    LOGGER.info(f"store_loc: {set_store_loc}")
    store = StoreInterface(store_loc=set_store_loc)


@pytest.mark.parametrize(
    "links", [(pytest.example_string_links), ({}), (pytest.example_links), (None)]
)
def test_setLinks(links, set_store_loc):
    """Tests if the actors links can be set to certain values."""

    act = Actor("test", set_store_loc)
    act.setLinks(links)
    assert act.links == links

@pytest.mark.parametrize(
    ("qc", "qs"),
    [
        ("comm", "sig"),
        (None, None),
        ("", ""),
        ("LINK", "LINK"),  # these are placeholder names (store is not setup)
    ],
)
def test_setCommLinks(example_links, qc, qs, init_actor, setup_store, set_store_loc):
    """Tests if commLinks can be added to the actor"s links."""

    if qc == "LINK" and qs == "LINK":
        qc = Link("L1", Actor("1", set_store_loc), Actor("2", set_store_loc))
        qs = Link("L2", Actor("3", set_store_loc), Actor("4", set_store_loc))
    act = init_actor
    act.setLinks(example_links)
    act.setCommLinks(qc, qs)

    example_links.update({"q_comm": qc, "q_sig": qs})
    assert act.links == example_links


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_setLinkIn(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the input queue."""

    act = init_actor
    act.setLinks(links)
    if links is not None:
        act.setLinkIn("input_q")
        expected.update({"q_in": "input_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("input_queue")


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_setLinkOut(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the output queue."""

    act = init_actor
    act.setLinks(links)
    if links is not None:
        act.setLinkOut("output_q")
        expected.update({"q_out": "output_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("output_queue")


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_setLinkWatch(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the watch queue."""

    act = init_actor
    act.setLinks(links)
    if links is not None:
        act.setLinkWatch("watch_q")
        expected.update({"q_watchout": "watch_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("input_queue")


def test_addLink(setup_store, set_store_loc):
    """Tests if a link can be added to the dictionary of links."""

    act = Actor("test", set_store_loc)
    links = {"1": "one", "2": "two"}
    act.setLinks(links)
    newName = "3"
    newLink = "three"
    act.addLink(newName, newLink)
    links.update({"3": "three"})

    # trying to check for two separate conditions while being able to
    # distinguish between them should an error be raised
    passes = []
    err_messages = []

    if act.getLinks()["3"] == "three":
        passes.append(True)
    else:
        passes.append(False)
        err_messages.append(
            "Error:\
            actor.getLinks()['3'] is not equal to \"three\""
        )

    if act.getLinks() == links:
        passes.append(True)
    else:
        passes.append("False")
        err_messages.append(
            "Error:\
            actor.getLinks() is not equal to the links dictionary"
        )

    err_out = "\n".join(err_messages)
    assert all(passes), f"The following errors occurred: {err_out}"


def test_getLinks(init_actor, example_string_links):
    """Tests if we can access the dictionary of links.

    TODO:
        Add more parametrized test cases.
    """

    act = init_actor
    links = example_string_links
    act.setLinks(links)

    assert act.getLinks() == {"1": "one", "2": "two", "3": "three"}


@pytest.mark.skip(
    reason="this is something we'll do later because\
                    we will subclass actor w/ watcher later"
)
def test_put(init_actor):
    """Tests if data keys can be put to output links.

    TODO:
        Ask Anne to take a look.
    """

    act = init_actor
    act.put()
    assert True


def test_run(init_actor):
    """Tests if actor.run raises an error."""
    act = init_actor
    with pytest.raises(NotImplementedError):
        act.run()


def test_changePriority(init_actor):
    """Tests if we are able to change the priority of an actor."""

    act = init_actor
    act.lower_priority = True
    act.changePriority()

    assert psutil.Process(os.getpid()).nice() == 19


def test_actor_connection(setup_store, set_store_loc):
    """Test if the links between actors are established correctly.

    This test instantiates two actors with different names, then instantiates
    a Link object linking the two actors. A string is put to the input queue of
    one actor. Then, in the other actor, it is removed from the queue, and
    checked to verify it matches the original message.
    """
    act1 = Actor("a1", set_store_loc)
    act2 = Actor("a2", set_store_loc)

    StoreInterface(store_loc=set_store_loc)
    link = Link("L12", act1, act2)

    act1.setLinkIn(link)
    act2.setLinkOut(link)

    msg = "message"

    act1.q_in.put(msg)

    assert act2.q_out.get() == msg


#=========================================
#
#
#
#
#           MANAGED ACTOR TESTS
#
#
#
#
#=========================================
@pytest.fixture
def init_managed_actor(init_actor):
    """ Fixture to create a managed actor.
    """
    mactor = ManagedActor(init_actor)
    yield mactor
    mactor = None


@pytest.fixture
def init_custom_managed_actor():
    """ Fixture to create and yield a managed actor that has been predefined.
    """
    pass
    #TODO: Implement this


#---------- DEFAULT -----------#

@pytest.mark.skip(reason="unfinished")
def test_default_ManagedActor_init(init_actor):
    pass


@pytest.mark.skip(reason="unfinished")
def test_default_ManagedActor_setup(init_actor):
    pass


@pytest.mark.skip(reason="unfinished")
def test_default_ManagedActor_run(init_actor):
    pass


@pytest.mark.skip(reason="unfinished")
def test_default_ManagedActor_run_step(init_actor):
    pass


@pytest.mark.skip(reason="unfinished")
def test_default_ManagedActor_stop(init_actor):
    pass

#---------- CUSTOM -----------#

@pytest.mark.skip(reason="unfinished")
def test_custom_ManagedActor_init():

    pass


@pytest.mark.skip(reason="unfinished")
def test_custom_ManagedActor_setup():
    pass


@pytest.mark.skip(reason="unfinished")
def test_custom_ManagedActor_run():
    pass


@pytest.mark.skip(reason="unfinished")
def test_custom_ManagedActor_run_step():
    pass


@pytest.mark.skip(reason="unfinished")
def test_custom_ManagedActor_stop():
    pass

#=========================================
#
#
#
#
#           ASYNC ACTOR TESTS
#
#
#
#
#=========================================

@pytest.mark.skip(reason="unfinished")
def test_AsyncActor_init():
    pass


@pytest.mark.skip(reason="unfinished")
def test_AsyncActor_run():
    pass


@pytest.mark.skip(reason="unfinished")
def test_AsyncActor_setup():
    pass


@pytest.mark.skip(reason="unfinished")
def test_AsyncActor_run_step():
    pass


@pytest.mark.skip(reason="unfinished")
def test_AsyncActor_stop():
    pass




