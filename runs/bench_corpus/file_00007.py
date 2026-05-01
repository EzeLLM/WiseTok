from typing import Union
from django.core.exceptions import ValidationError
from django.db import models
from django.contrib.auth.models import User
from django.db.models.fields.related import ForeignKey

# problem it allows creating multiple threads ex p1-p2 and p2-p1 are considerd different


class SingleChatThread(models.Model):
    users = models.ManyToManyField(User, related_name='my_threads')

    def get_messages(self, x: int = 10):
        return self.messages.all()[:x]  # type: ignore


class GroupChatThread(models.Model):
    users = models.ManyToManyField(User, related_name='my_group_threads')

    def get_messages(self, x: int = 10):
        return self.messages.all()[:x]  # type: ignore


class Message(models.Model):
    thread = models.ForeignKey(
        SingleChatThread, related_name='messages', on_delete=models.CASCADE)
    author = models.ForeignKey(
        User, related_name='chat_messages', on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.author.username
