from django.shortcuts import render
from  .forms import contactform
from django.http.response import HttpResponse
def home(request):
    if request.method == "POST":
        xyz = contactform(request.POST)
        if xyz.is_valid():
            return HttpResponse("valid Form")
        else:
            return HttpResponse("Invalid Form")
    else:
        xyz=contactform()
        return render(request,'simplecontact.html',{'abc':xyz})

