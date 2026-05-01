"""
Scheduled jobs to run in background.
- Penalise overdue tasks (24hrs)
- Hard reset at 00:00 on Mondays (system timezone)
"""

import webapp

import time
import random

RESOURCES = ["water", "food", "metal"]


def penalise_overdue():
    """
    Cronjob to check and penalise overdue tasks every hour.
    Change penalised tasks to completed for now.
    TODO: Set an overdue attribute and render differently in frontend.
    """
    pb_db = webapp.pb.database()
    all_users_dict = pb_db.child('users').get().val()
    for uid in all_users_dict.keys():
        user = all_users_dict[uid]
        group = user["group"]
        user_tasks_dict = user["tasks"]
        for task in user_tasks_dict.keys():
            if not user_tasks_dict[task]["completed"] and is_overdue(user_tasks_dict[task]["timestamp"]):
                remove_random_resources(pb_db, group)
                pb_db.child("users").child(uid).child(task).update({"completed": True})


def hard_reset():
    """
    Cronjob to remove all tasks (completed or not) on Mondays.
    """
    pb_db = webapp.pb.database()
    all_users_dict = pb_db.child('users').get().val()
    for uid in all_users_dict.keys():
        user = all_users_dict[uid]
        user_tasks_dict = user["tasks"]
        for task in user_tasks_dict.keys():
            pb_db.child('users').child(uid).child("tasks").child(task).remove()


def is_overdue(timestamp):
    """
    Return if is timestamp is 24hrs away.
    """
    return (time.time() - timestamp) > 86400


def remove_random_resources(pb_db, group, unit=5):
    """
    Remove 5 units of random resources from group, do nothing if the resource reaches zero.
    """
    resource_to_remove = RESOURCES[random.randint(0, 2)]
    quantity = min(0, pb_db.child("groups").child(group).child("resources").child(resource_to_remove).get().val() - unit)
    pb_db.child("groups").child(group).child("resources").update({resource_to_remove: quantity})