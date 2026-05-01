from rest_framework.response import Response
from rest_framework.views import status
from intranet.models import Authors, Publisher, Suggestion, Borrowers, ModeratorReasons, Biblio, Genre
from intranet.models import Authors, CorporateAuthor, Language, Publisher
from intranet.models import suggestions_status_choices, itemtype_choices
import datetime
import re
import decimal
from accounts.models import User, Profile

def validate_request_data_authors(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        firstname = args[0].request.data.get("firstname", "")
        lastname = args[0].request.data.get("lastname", "")
        if not firstname or not lastname:
            return Response(
                data={
                    "message": "Both firstname and lastname are required to add an author"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        else:
            firstname = firstname.strip()
            re.sub(' +', ' ',firstname)
            lastname = lastname.strip()
            re.sub(' +', ' ',lastname)
            authors = Authors.objects.filter(firstname__iexact=firstname,lastname__iexact=lastname)
            if len(authors) > 0:
                return Response(
                    data={
                        "message": "A record with that author already exists"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        return fn(*args, **kwargs)
    return decorated

def validate_request_data_publishers(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        data = args[0].request.data
        messages=[]
        if isinstance(data,list):
            i=0
            while i<len(data):
               name = data[i].get("name")
               if not name:
                  messages.append("Record({}): Name is required".format(i+1))
               name = name.strip()
               re.sub(' +', ' ',name)
               publishers = Publisher.objects.filter(name__iexact=name)
               if len(publishers) > 0:
                  messages.append("Record({}) : An entry with that Name '{}' already exists".format(i+1,name))
               i=i+1
        else: 
           name=data.get('name')
           name = name.strip()
           re.sub(' +', ' ',name)
           publishers = Publisher.objects.filter(name__iexact=name)
           if len(publishers) > 0:
               messages.append("An entry with that Name '{}' already exists".format(name))
        if len(messages)>0:
            return Response(
                data={
                    "message": messages,
                },
                status=status.HTTP_400_BAD_REQUEST
            )
                
        return fn(*args, **kwargs)
    return decorated

#put a check on max number of suggestions by an individual - limit 2
def validate_request_data_post_suggestions(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        profile_id = User.objects.get(email=args[0].request.user).profile.id
        
        user_id=None
        try:
           user_= Borrowers.objects.get(borrower=profile_id)
           if user_:
              user_id=user_.id
        except:
              pass
           
        if user_id==None:
            return Response(
               data={
                   "message": "You are not a library borrower. You can suggest only then."
               },
               status=status.HTTP_400_BAD_REQUEST
            )
           
        numberOfSuggestions = 0
        if user_:
            numberOfSuggestions = Suggestion.objects.filter(suggestedby=user_id,status='ASKED').count()
        if numberOfSuggestions>1:
            return Response(
                data={
                        "message": "You are not allowed to give more than 2 suggestions at a time"
                    },
                    status=status.HTTP_400_BAD_REQUEST
            )
        title = args[0].request.data.get("title", "")
        if not title:
            return Response(
                data={
                    "message": "Title is required for the suggestion to be added"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        isbn = args[0].request.data.get("isbn", "")
        if isbn:
            if len(isbn)>=10 and re.match(r'(\d{10}|\d{12}[0-9X])',isbn,re.M|re.I):
                pass
            else:
                return Response(
                    data={
                        "message": "Invalid ISBN"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        copyrightdate = args[0].request.data.get("copyrightdate", "")              
        if copyrightdate:
            yr = str(copyrightdate)
            year = re.match(r'(^\d{4}$)',yr)
            if year:
                pass   
            else:
                return Response(
                    data={
                        "message": "Invalid Copyrightdate"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        price = args[0].request.data.get("price", "")              
        if price:
            price_ = str(price)
            price = re.search(r'(\d{1,4}|\d{1,4}\.\d\d)',price_)
            if price:
                pass   
            else:
                return Response(
                    data={
                        "message": "Invalid price. Should be between 0 and 9999. No decimals allowed."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        total = args[0].request.data.get("total", "")              
        if total:
            total_ = str(total)
            total = re.search(r'(\d{1,5}|\d{1,5}\.\d\d)',total_)
            if total:
                pass   
            else:
                return Response(
                    data={
                        "message": "Invalid total amount. Should be between 0 and 99999. No decimals allowed."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        return fn(*args, **kwargs)
        
    return decorated

def validate_request_data_put_suggestions(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        suggestion = Suggestion.objects.get(pk=kwargs["pk"])
        if suggestion:
            suggestedby = suggestion.suggestedby
        else:
            return Response(
                data={
                    "message": "No such suggestion exists"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        request_user = User.objects.get(email=args[0].request.user)
        profile_id = request_user.profile.id
        user_ = Borrowers.objects.get(borrower=profile_id)
        if not (request_user.staff==True or suggestedby == user_): 
            return Response(
                data={
                    "message": "Only staff or the owner can edit/delete/moderate suggestion"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not user_:
            return Response(
                data={
                    "message": "You need to be a library borrower to edit/delete suggestion"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        '''
        #suggested_user = Borrowers.objects.get(borrower=suggestedby.borrower)
        existing_title = suggestion.title
        request_title = args[0].request.data.get("title","")
        if request_title!="" and user_!=suggestedby:
            return Response(
                data={
                    "message": "Only the owner of suggestion can change the title of the Suggestion"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        '''
        s_status = args[0].request.data.get("status", "")
        if s_status =="":
           s_status = suggestion.status
        if s_status=="":
            return Response(
                data={
                    "message": "Suggestion status can't be left empty"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        else:
          s_status = s_status.upper()
          status_ = []
          for choice in suggestions_status_choices:
              status_.append(choice[0])
          if not s_status in status_:
              return Response(
                data={
                    "message": "Invalid suggestion status. Should be one of '{}'".format(status_)
                },
                status=status.HTTP_400_BAD_REQUEST
              )
          if user_==suggestedby and (not s_status=='ASKED'):
              return Response(
                  data={
                      "message": "You can't moderate your own suggestion. Leave the status as 'ASKED'."
                  },
                  status=status.HTTP_400_BAD_REQUEST
              )  
        isbn = args[0].request.data.get("isbn", "")
        if isbn:
            if re.match(r'(^\d{10}$|^\d{12}[0-9X]$)',isbn,re.M|re.I):
                pass
            else:
                return Response(
                    data={
                        "message": "Invalid ISBN"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        copyrightdate = args[0].request.data.get("copyrightdate", "")              
        if copyrightdate:
            year = re.match(r'(^\d{4}$)',copyrightdate)
            if year:
                copyrightdate = year   
            else:
                return Response(
                    data={
                        "message": "Invalid Copyrightdate"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        price = args[0].request.data.get("price", "")              
        if price:
            price_ = str(price)
            price = re.search(r'(\d{1,4}|\d{1,4}\.\d\d)',price_)
            if price:
                pass   
            else:
                return Response(
                    data={
                        "message": "Invalid price. Should be between 0 and 9999. No decimals allowed."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        total = args[0].request.data.get("total", "")              
        if total:
            total_ = str(total)
            total = re.search(r'(\d{1,5}|\d{1,5}\.\d\d)',total_)
            if total:
                pass   
            else:
                return Response(
                    data={
                        "message": "Invalid total amount. Should be between 0 and 99999. No decimals allowed."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        reason = args[0].request.data.get("reason", "")
        reasonids=[]
        if reason:
           reason=int(reason)
           reasons = ModeratorReasons.objects.all()               
           for reason_ in reasons:
               reasonids.append(reason_.id)
           if len(reasonids)>0 and reason not in reasonids:
               message = "Invalid reason. It should be a number and should be one of '{}'.".format(reasonids)
               return Response(
                   data={
                       "message": message,
                   },
                   status=status.HTTP_400_BAD_REQUEST
               )
        return fn(*args, **kwargs)
    return decorated



def validate_request_data_delete_suggestions(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        suggestion = Suggestion.objects.get(pk=kwargs["pk"])
        if suggestion:
            suggestedby = suggestion.suggestedby
        else:
            return Response(
                data={
                    "message": "No such suggestion exists"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        request_user = User.objects.get(email=args[0].request.user)
        profile_id = request_user.profile.id
        user_ = Borrowers.objects.get(borrower=profile_id)
        if not (request_user.staff==True or suggestedby == user_): 
            return Response(
                data={
                    "message": "Only staff or the owner can edit/delete/moderate suggestion"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        return fn(*args, **kwargs)
    return decorated


def validate_request_data_biblio(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        isbn = args[0].request.data.get("isbn", "")
        if isbn:
            if len(isbn)>=10 and re.search(r'(^\d{10}$s|^\d{12}[0-9x]$)',isbn,re.M|re.I):
                pass
            else:
                return Response(
                    data={
                        "message": "Invalid ISBN"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        copyrightdate = args[0].request.data.get("copyrightdate", "")              
        if copyrightdate:
            yr = str(copyrightdate)
            if re.match(r'(^\d{4}$)',yr):
                pass   
            else:
                return Response(
                    data={
                        "message": "Invalid Copyrightdate. It should be year of publication."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
                
                return fn(*args, **kwargs)

        itemtype = args[0].request.data.get("itemtype", "")
        if itemtype:
           itemtypes = []
           for itemtype_ in itemtype_choices:
               itemtypes.append(itemtype_[0])   
           if itemtype.upper()  not in itemtypes:
               message = "Invalid itemtype: It should be one of '{}'.".format(itemtypes)
               return Response(
                   data={
                       "message": message,
                   },
                   status=status.HTTP_400_BAD_REQUEST
               ) 
        corporateauthor = args[0].request.data.get("corporateauthor", "")
        corpauthor_ids=[]
        if corporateauthor:
           corporateauthor=int(corporateauthor)
           corpauthors = CorporateAuthor.objects.all()               
           for corpauthor_ in corpauthors:
               corpauthor_ids.append(corpauthor_.id)
           if len(corpauthor_ids)>0 and corporateauthor not in corpauthor_ids:
               message = "Invalid corporate author. It should be a number and should be one of '{}'.".format(corpauthor_ids)
               return Response(
                   data={
                       "message": message,
                   },
                   status=status.HTTP_400_BAD_REQUEST
               )
        publisher = args[0].request.data.get("publisher", "")
        if publisher:
           if not Publisher.objects.get(pk=int(publisher)):
               message = 'Publisher[{}] is invalid'.format(publisher)
               return Response(
                   data={
                       "message": message,
                   },
                   status=status.HTTP_400_BAD_REQUEST
               )
        language = args[0].request.data.get("language", "")
        if language:
           if not Language.objects.get(pk=int(language)):
               message = 'Language[{}] is invalid'.format(language)
               return Response(
                   data={
                       "message": message,
                   },
                   status=status.HTTP_400_BAD_REQUEST
               ) 
        authors = args[0].request.data.get("authors", "")
        message = []
        if len(authors)>0:
           for author in authors:
               if not Authors.objects.get(pk=int(author)):
                   message.append('Author[{}] is invalid'.format(author))      
           if len(message)>0:               
               return Response(
                   data={
                       "message": message,
                   },
                   status=status.HTTP_400_BAD_REQUEST
               )
        
        genre = args[0].request.data.get("genre", "")
        message = []
        if len(genre)>0:
           for genre_ in genre:
               if not Genre.objects.get(pk=int(genre_)):
                   message.append('Genre[{}] is invalid'.format(genre))      
           if len(message)>0:               
               return Response(
                   data={
                       "message": message,
                   },
                   status=status.HTTP_400_BAD_REQUEST
               )
        
        return fn(*args, **kwargs)        
    return decorated

from rest_framework.response import Response
from rest_framework.views import status
from intranet.models import Borrowers, PatronPhotos
import re

def validate_request_data_borrowers(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        data = args[0].request.data
        messages=[]
        if isinstance(data,list):
            i=0
            while i<len(data):
               firstname = data[i].get("firstname")
               surname = data[i].get("surname")
               if not (firstname or surname)  :
                  messages.append("Record({}): Both First & Surname are required".format(i+1))
               firstname = firstname.strip()
               surname = surname.strip()
               re.sub(' +', ' ',firstname)
               re.sub(' +', ' ',surname)
               patrons = Borrowers.objects.filter(firstname__iexact=firstname, surname__iexact=surname)
               if len(patrons) > 0:
                  messages.append("Record({}) : An entry with that Name '{}' already exists".format(i+1,firstname+" "+surname))
               
               cardnumber = data[i].get("cardnumber")
               if not (cardnumber)  :
                  messages.append("Record({}): Cardnumber is required".format(i+1))
               cardnumber = cardnumber.strip()
               cardnumber = cardnumber.upper()
               re.sub(' +', ' ',cardnumber)
               patrons = Borrowers.objects.filter(cardnumber__iexact=cardnumber)
               if len(patrons) > 0:
                  messages.append("Record({}) : An entry with that Cardnumber '{}' already exists".format(i+1,cardnumber))
               
               email = data[i].get("email")
               if not (email):
                  messages.append("Record({}): Email is required".format(i+1))
               email = email.strip()
               re.sub(' +', ' ',email)
               patrons = Borrowers.objects.filter(email__iexact=email)
               if len(patrons) > 0:
                  messages.append("Record({}) : An entry with that email '{}' already exists".format(i+1,email))
               dateexpired = data[i].get("dateexpired")
               if not (dateexpired):
                  messages.append("Record({}): Expiry date is required".format(i+1))
               category = data[i].get("category")
               if not (category):
                  messages.append("Record({}): Category is required".format(i+1))   
               department = data[i].get("department")
               if not (department):
                  messages.append("Record({}): Department is required".format(i+1))   
               designation = data[i].get("designation")
               if not (designation):
                  messages.append("Record({}): Designation is required".format(i+1))  
               borrower = data[i].get('borrower')
               if not (borrower or email):
                  messages.append("Record({}): Both email and Patron file are required".format(i+1))
               else:
                  profile_id = User.objects.get(email=email).profile.id
                  user_= Borrowers.objects.get(borrower=profile_id)
                  if user_:
                      messages.append("Record({}):A borrower with that profile already exists".format(i+1))
               i=i+1
        else: 
           firstname = data.get("firstname")
           surname = data.get("surname")
           firstname = firstname.strip()
           surname = surname.strip()
           firstname = firstname.strip()
           re.sub(' +', ' ',firstname)
           re.sub(' +', ' ',surname)
           patrons = Borrowers.objects.filter(firstname__iexact=firstname,surname__iexact=surname)
           if len(patrons) > 0:
               messages.append("An entry with that Name '{}' already exists".format(firstname+" " +surname))
           email = data.get("email")
           if not(email):
               messages.append("Email is required")
               email = email.strip()
               re.sub(' +', ' ',email)
               email_ = Borrowers.objects.filter(email__iexact=email)
               if len(email_)>0:
                   messages.append("An borrower with that email '{}' already exists".format(email))
           cardnumber = data.get("cardnumber")
           if not (cardnumber):
               messages.append("Cardnumber is required")
               cardnumber = cardnumber.strip()
               cardnumber = cardnumber.upper()
               re.sub(' +', ' ',cardnumber)
               patrons = Borrowers.objects.filter(cardnumber__iexact=cardnumber)
               if len(patrons) > 0:
                  messages.append("A borrower with that Cardnumber '{}' already exists".format(cardnumber))
           dateexpiry = data.get("dateexpiry")
           if not (dateexpiry):
               messages.append("Expiry date must be entered")  
           category = data.get("category")
           if not (category):
               messages.append("Category must be selected")   
           department = data.get("department")
           if not (department):
               messages.append("Department must be selected")   
           designation = data.get("designation")
           if not (designation):
               messages.append("Designation must be selected")    
           profile_id = None
           try:
               profile_id = Users.objects.get(pk=data.get('borrower')).profile.id
           except:
               pass 
           if not (profile_id or email):
               messages.append("Email is required and Patron Profile must be selected")
           else:
               user_=None
               try:
                  user_= Borrowers.objects.get(borrower=profile_id)
               except:
                  pass
               if user_:
                    messages.append("A borrower with that profile already exists")
               else:
                   profile_email=None
                   try:
                      profile_email = User.objects.get(pk=data.get('borrower')).email
                   except:
                      pass
                   if email and (not email==profile_email):
                       messages.append("Email entered and Profile email are not same")
                       
                       return Response(
                           data={
                               "message": messages,
                           },
                           status=status.HTTP_400_BAD_REQUEST
                       )

        if len(messages)>0:
            return Response(
                data={
                    "message": messages,
                },
                status=status.HTTP_400_BAD_REQUEST
            )
                
        return fn(*args, **kwargs)
    return decorated

#import mimetypes
import tempfile
import os
import magic

def file_path_mime(file_path):
    mime = magic.from_file(file_path, mime=True)
    return mime

def check_in_memory_mime(in_memory_file):
    mime = magic.from_buffer(in_memory_file.read(), mime=True)
    return mime

valid_mime_types = ('image/jpg','image/jpeg','image/png')

def validate_request_data_patronphotos(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        data = args[0].request.data
        imageurl = data.get("imageurl")
        patron_id = data.get("patron")
        mime = magic.Magic(mime=True)
        file_mime_type = check_in_memory_mime(args[0].request.FILES['imageurl'])
        print("file_mime_type = ",file_mime_type)
        if file_mime_type not in valid_mime_types: 
           return Response(
              data={
                 "message": "Only png and jpg mime type images are allowed",
              },
              status=status.HTTP_400_BAD_REQUEST
           )
        try:
            if PatronPhotos.objects.get(patron=patron_id): 
                return Response(
                    data={
                        "message": "A photo for the patron with that id: %d already exists" %  patron_id
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            if not (imageurl or patron_id)  :
                return Response(
                    data={
                        "message":"Both Patron & Image Url are required",
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
        except:
            pass 
                
        return fn(*args, **kwargs)
    return decorated

