from django.conf.urls import url
from django.contrib import admin
from MyBlog.views import *

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^myBlogs/$',myBlogs),
    url(r'^Blogs/$',Blogs),
]
