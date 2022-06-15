import pytest
import subprocess
from improv.actor import Actor as actor
from improv.store import Limbo as limbo

#set global variables
pytest.sample_links = {'1': "one", '2': "two", '3': "three"}

@pytest.fixture
def setup_store():
    p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)],\
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    yield p
    p.kill()

@pytest.mark.parametrize("attribute, expected", [("q_watchout", None), (
    "name", ""), ("links", {}), ("done", False), ("lower_priority", False), (
    "q_in", None), ("q_out", None)])
def test_default_init(attribute, expected):
    """ Tests if the default init attributes are as expected.
    """

    act = actor("")
    atr = getattr(act, attribute)

    assert atr == expected

def test_repr_default_initialization():
    """ Test if the actor representation has the right dict keys.
    """

    act = actor("Alan")
    rep = act.__repr__()
    assert rep == "Alan: dict_keys([])"

def test_setStore(setup_store):
    """ Tests if the store is started and linked with the actor.
    """

    act = actor("Acquirer")
    lmb = limbo()
    act.setStore(lmb.client)
    assert act.client is lmb.client

@pytest.mark.parametrize("links, expected", [
    (pytest.sample_links,\
    pytest.sample_links),\
    ({}, {}), (None, None)])
def test_setLinks(links, expected):
    """ Tests if the actors links can be set to certain values.
    """

    act = actor("test")
    act.setLinks(links)
    assert act.links == expected

@pytest.mark.parametrize("links, qc, qs, expected", [
    (pytest.sample_links, "comm", "sig", {
    '1': "one", '2': "two", '3': "three", "q_comm": "comm", "q_sig": "sig"}),
    (pytest.sample_links, None, None, {'1': "one", '2': "two", '3': "three",
    "q_comm": None, "q_sig": None})])
def test_setCommLinks(links, qc, qs, expected):
    """ Tests if commLinks can be added to the actor's links.
    """

    act = actor("Test")
    act.setLinks(links)
    act.setCommLinks(qc, qs)
    assert act.links == expected

@pytest.mark.skip(reason="this test is unfinished")
def test_setLinkIn():
    assert True

@pytest.mark.skip(reason="this test is unfinished")
def test_setLinkOut():
    assert True

@pytest.mark.skip(reason="this test is unfinished")
def test_setLinkWatch():
    assert True

def test_addLink(setup_store):
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

@pytest.mark.skip(reason="this test is unfinished")
def test_getLinks():
    assert True

@pytest.mark.skip(reason="this test is unfinished")
def test_setup():
    assert True

@pytest.mark.skip(reason="this test is unfinished")
def test_put():
    assert True

@pytest.mark.skip(reason="this test is unfinished")
def test_run():
    assert True

@pytest.mark.skip(reason="this test is unfinished")
def test_changePriority():
    assert True


