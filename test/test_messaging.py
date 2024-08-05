import improv.messaging


def test_actor_state_msg():
    name = "test name"
    status = "test status"
    port = 12345
    info = "test info"
    msg = improv.messaging.ActorStateMsg(name, status, port, info)
    assert msg.actor_name == name
    assert msg.status == status
    assert msg.nexus_in_port == port
    assert msg.info == info


def test_actor_state_reply_msg():
    name = "test name"
    status = "test status"
    info = "test info"
    msg = improv.messaging.ActorStateReplyMsg(name, status, info)
    assert msg.actor_name == name
    assert msg.status == status
    assert msg.info == info


def test_actor_signal_msg():
    name = "test name"
    signal = "test signal"
    info = "test info"
    msg = improv.messaging.ActorSignalMsg(name, signal, info)
    assert msg.actor_name == name
    assert msg.signal == signal
    assert msg.info == info


def test_actor_signal_reply_msg():
    name = "test name"
    status = "test status"
    signal = "test_signal"
    info = "test info"
    msg = improv.messaging.ActorSignalReplyMsg(name, signal, status, info)
    assert msg.actor_name == name
    assert msg.status == status
    assert msg.info == info
    assert msg.signal == signal


def test_broker_info_msg():
    name = "test name"
    pub_port = 54321
    sub_port = 12345
    info = "test info"
    msg = improv.messaging.BrokerInfoMsg(name, pub_port, sub_port, info)
    assert msg.name == name
    assert msg.pub_port == pub_port
    assert msg.sub_port == sub_port
    assert msg.info == info


def test_broker_info_reply_msg():
    name = "test name"
    status = "test status"
    info = "test info"
    msg = improv.messaging.BrokerInfoReplyMsg(name, status, info)
    assert msg.name == name
    assert msg.status == status
    assert msg.info == info


def test_log_info_msg():
    name = "test name"
    pull_port = 54321
    pub_port = 12345
    info = "test info"
    msg = improv.messaging.LogInfoMsg(name, pull_port, pub_port, info)
    assert msg.name == name
    assert msg.pub_port == pub_port
    assert msg.pull_port == pull_port
    assert msg.info == info


def test_log_info_reply_msg():
    name = "test name"
    status = "test status"
    info = "test info"
    msg = improv.messaging.LogInfoReplyMsg(name, status, info)
    assert msg.name == name
    assert msg.status == status
    assert msg.info == info


def test_harvester_info_msg():
    name = "test name"
    info = "test info"
    msg = improv.messaging.HarvesterInfoMsg(name, info)
    assert msg.name == name
    assert msg.info == info


def test_harvester_info_reply_msg():
    name = "test name"
    status = "test status"
    info = "test info"
    msg = improv.messaging.HarvesterInfoReplyMsg(name, status, info)
    assert msg.name == name
    assert msg.status == status
    assert msg.info == info
