from notify_run import Notify


def sendnotificate(message):
    notify = Notify()
    notify.send(message)
