from django.contrib import admin
from main.models import Patient, Address
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

# Register your models here.
admin.site.register(Patient)
admin.site.register(Address)
# 
# class PatientInline(admin.StackedInline):
#     model = Patient
#     canDelete = False
#     verbose_name_plural = "patients"
#
# class UserAdmin(BaseUserAdmin):
#     inlines = (PatientInline, )
#
# admin.site.unregister(User)
# admin.site.register(User, UserAdmin)
