# my_project/urls.py
# from django.urls import path, include

from django.conf.urls import url
from django.conf.urls import include
from django.contrib import admin
from django.contrib.auth import views
from django.views.generic.base import TemplateView # new
from django.urls import path

#urlpatterns = [
#    url(r'^admin/', admin.site.urls),
#    url(r'^polls/', include('polls.urls')),
#    path('', TemplateView.as_view(template_name='home.html'), name='home'),
#]

# url(r'^profiles/home', views.home, name='home'),
# url(r'^accounts/login/$', views.login, name='login'),
# url(r'^accounts/logout/$', views.logout, name='logout'),

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    url(r'^polls/', include('polls.urls')),
]
