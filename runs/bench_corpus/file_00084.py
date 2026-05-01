from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _

# Create your models here.


class User(AbstractUser):
    pass


class FundCompany(models.Model):
    name = models.CharField(
        _('name'), help_text='名称', max_length=64)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'fund_company'
        ordering = ['-id']


class FundManager(models.Model):
    name = models.CharField(
        _('name'), help_text='姓名', max_length=64)
    working_seniority = models.DateField(
        _('working seniority'), help_text='从业时间')
    fund_company = models.ForeignKey(
        FundCompany,
        related_name='ref_fund_company',
        default=None,
        blank=True,
        null=True,
        on_delete=models.SET_NULL)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'fund_manager'
        ordering = ['-id']


class Fund(models.Model):
    name = models.CharField(
        _('name'), help_text='名称', unique=True, max_length=64)
    code = models.CharField(
        _('name'), help_text='基金代码', unique=True, max_length=32)
    type = models.CharField(
        _('type'), help_text='基金类型', max_length=32)
    scale = models.FloatField(
        _('scale'), help_text='基金规模:亿元')
    found_date = models.DateField(
        _('found date'), help_text='成立日')
    managers = models.ManyToManyField(FundManager, default=None, blank=True)
    custodian = models.ForeignKey(
        FundCompany,
        related_name='ref_custodian',
        default=None,
        blank=True,
        null=True,
        on_delete=models.SET_NULL)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = _('fund')
        db_table = 'fund'


class FundAccount(models.Model):
    date = models.DateField(
        _('date'), help_text='日期')
    principal = models.FloatField(
        _('principal'), help_text='本金')
    amount = models.FloatField(
        _('amount'), help_text='金额')
    fund_net = models.FloatField(
        _('fund net'), help_text='基金净值')
    share = models.FloatField(
        _('fund share'), help_text='基金份额')
    fund = models.ForeignKey(
        Fund,
        related_name='ref_fund',
        default=None,
        blank=True,
        null=True,
        on_delete=models.PROTECT)

    class Meta:
        verbose_name = _('fund account')
        db_table = 'fund_account'
