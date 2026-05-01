# Register your models here.

from django.contrib import admin
from .models import Post
from .models import Customer
from .models import Customergroup
from .models import Nation
from .models import Legalform
from .models import Currency
from .models import State
from .models import Address
from .models import Contact


admin.site.register(Post)
admin.site.register(Customer)
admin.site.register(Customergroup)
admin.site.register(Nation)
admin.site.register(Legalform)
admin.site.register(Currency)
admin.site.register(State)
admin.site.register(Address)
admin.site.register(Contact)
