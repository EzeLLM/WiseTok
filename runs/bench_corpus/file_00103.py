from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
from .models import User
  # the index function is called when root is visited
  # Create your views here.
  
def index(request):
    #anything you did in the shell you can insert here
    return render(request,'users/user.html', { "users": User.objects.all()})


def new(request):
    return render(request,'users/new.html')

def create(request):
    errors = User.objects.basic_validator(request.POST)
    if len(errors):
        for key, value in errors.iteritems():
            messages.error(request, value)
        return redirect('/users/new/')
    else:
        User.objects.create(first_name=request.POST['first_name'], last_name=request.POST['last_name'], email=request.POST['email'])
    return render(request,'users/user.html', { "users": User.objects.all()})
    


def show(request, id):
    return render(request,'users/show.html', { "one_user": User.objects.get(id=id)})


def edit(request, id):
    return render(request,'users/edit.html', { "one_user": User.objects.get(id=id)})


def update(request, id):
    errors = User.objects.basic_validator(request.POST)
    if len(errors):
        for key, value in errors.iteritems():
            messages.error(request, value)
        return redirect('/users/'+id+'/edit/')
    else:
        user = User.objects.get(id=id)
        user.first_name = request.POST['first_name']
        user.last_name = request.POST['last_name']
        user.email = request.POST['email']
        user.save()
        return render(request,'users/user.html', { "users": User.objects.all()})



def remove(request, id):
    b=User.objects.get(id=id)
    b.delete()
    return redirect('/users/')