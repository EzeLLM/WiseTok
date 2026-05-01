import pygame
from pygame import gfxdraw
pygame.init()
win = pygame.display.set_mode((500,500))
pygame.display.set_caption('Mid-Point Circle Drawing Program')
win.fill((0,0,0))
color=(255,255,255)


def drawPoints(x,y):
    gfxdraw.pixel(win,x+a+250,y+b+250,color)
def MidPointEllipse():
    p=(ry**2)+(rx**2)/4-(rx**2)*ry
    x=0
    y=ry
    drawPoints(x,y)
    while ((2*x*ry**2)<(2*y*rx**2)):
        if (p>=0):
            x=x+1
            y=y-1
            p=p+2*x*(ry**2)+(ry**2)-2*(rx**2)*y
           
        else:
            x=x+1
            y=y
            p=p+2*x*(ry**2)+(ry**2)
       # print('p = ',p,' (x , y) = ' , x , ' ' , y)
        drawPoints(x,y)
        drawPoints(x,-y)
        drawPoints(-x,-y)
        drawPoints(-x,y)
 

    
    p=rx**2+((ry**2)/4)-(rx*(ry**2))
    x=rx
    y=0
    drawPoints(x,y)
    while ((2*x*ry**2)>(2*y*rx**2)):
        if (p>=0):
            x=x-1
            y=y+1
            p=p-2*x*(ry**2)+rx**2+(2*(rx**2))*y
        else:
            x=x
            y=y+1
            p=p+2*y*(rx**2)+(rx**2)
        #print('p = ',p,' (x , y) = ' , x , ' ' , y)
        drawPoints(x,y)
        drawPoints(x,-y)
        drawPoints(-x,-y)
        drawPoints(-x,y)
    
   


a,b,c,d = input('Enter center and radius of the ellipse').split()
a = int(a)
b = int(b)
rx = int(c)
ry = int(d)
MidPointEllipse()
temp = ry
ry=rx
rx=temp
MidPointEllipse()
pygame.display.update()
while 1:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
