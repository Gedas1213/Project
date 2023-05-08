from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

# Create your models here.


class UserAccount(models.Model):
    account_number = models.IntegerField()
    account_password = models.CharField(max_length=50)
    server = models.CharField(max_length=100)
    owner = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return f'{self.account_number} {self.account_password} {self.server}'

    def get_absolute_url(self):
        return reverse('myaccounts', args=[str(self.id)])
