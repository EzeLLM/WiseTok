"""Transporte2 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import patterns, include, url
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib import admin
from django.conf import settings

admin.autodiscover()

urlpatterns = patterns('',
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'cliente.views.inicio', name='inicio'),
    url(r'^cliente/nuevo/$','cliente.views.clienteCreation' , name='cliente_nuevo'),
    url(r'^cliente/listado/$','cliente.views.clienteList' , name='cliente_listado'),
    url(r'^cliente/(?P<id_Cedula>\d+)$','cliente.views.clienteUpdate', name='cliente_detalle'),
    url(r'^cliente_e/(?P<id_Cedula>\d+)$','cliente.views.clienteDelete', name='cliente_borrar'),
    url(r'^ruta/nuevo/$','ruta.views.rutaCreation' , name='ruta_nuevo'),
    url(r'^ruta/listado/$','ruta.views.rutaList' , name='ruta_listado'),
    url(r'^ruta/(?P<id_Codigo>\d+)$','ruta.views.rutaUpdate', name='ruta_detalle'),
    url(r'^ruta_e/(?P<id_Codigo>\d+)$','ruta.views.rutaDelete', name='ruta_borrar'),
    url(r'^cupo/nuevo/$','cupo.views.cupoCreation', name='cupo_nuevo'),
    url(r'^cupo/listado/$','cupo.views.cupoList', name='cupo_listado'),
    url(r'^cupo/(?P<id_Cupo>\d+)$','cupo.views.cupoUpdate', name='cupo_detalle'),
    url(r'^cupo_e/(?P<id_Cupo>\d+)$','cupo.views.cupoDelete', name='cupo_borrar'),
)
