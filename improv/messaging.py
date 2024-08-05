class ActorStateMsg:
    def __init__(self, actor_name, status, nexus_in_port, info):
        self.actor_name = actor_name
        self.status = status
        self.nexus_in_port = nexus_in_port
        self.info = info


class ActorStateReplyMsg:
    def __init__(self, actor_name, status, info):
        self.actor_name = actor_name
        self.status = status
        self.info = info


class ActorSignalMsg:
    def __init__(self, actor_name, signal, info):
        self.actor_name = actor_name
        self.signal = signal
        self.info = info


class ActorSignalReplyMsg:
    def __init__(self, actor_name, signal, status, info):
        self.actor_name = actor_name
        self.signal = signal
        self.status = status
        self.info = info


class BrokerInfoMsg:
    def __init__(self, name, pub_port, sub_port, info):
        self.name = name
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.info = info


class BrokerInfoReplyMsg:
    def __init__(self, name, status, info):
        self.name = name
        self.status = status
        self.info = info


class LogInfoMsg:
    def __init__(self, name, pull_port, pub_port, info):
        self.name = name
        self.pull_port = pull_port
        self.pub_port = pub_port
        self.info = info


class LogInfoReplyMsg:
    def __init__(self, name, status, info):
        self.name = name
        self.status = status
        self.info = info


class HarvesterInfoMsg:
    def __init__(self, name, info):
        self.name = name
        self.info = info


class HarvesterInfoReplyMsg:
    def __init__(self, name, status, info):
        self.name = name
        self.status = status
        self.info = info
