import cng as c

__fen = None
__sousFen = None
__rect1 = None
__rect2 = None
x1 = None
y1 = None
x2 = None
y2 = None


######## mouvements du rectangle ########


def bougerHaut():

	global __rect1
	global __rect2
	
	c.obj_move(__rect1, 0, 5)
	c.obj_move(__rect2, 0, 5*(350/800))
	
def bougerBas():

	global __rect1
	global __rect2
	
	c.obj_move(__rect1, 0, -5)
	c.obj_move(__rect2, 0, -5*(350/800))
	
def bougerGauche():

	global __rect1
	global __rect2
	
	c.obj_move(__rect1, -5, 0)
	c.obj_move(__rect2, -5*(350/800), 0)
	
def bougerDroite():

	global __rect1
	global __rect2
	
	c.obj_move(__rect1, 5, 0)
	c.obj_move(__rect2, 5*(350/800), 0)
	

##############################

	


def main():

	global __fen
	global __sousFen
	
	global __rect1
	global __rect2
	global x1, x1, y2, y2

	__fen = c.init_window( "fenetre ex1 !", 1920, 1020, "white")
	
	x1 = 480
	y1 = 360
	x2 = 720 + 480
	y2 = 600 + 360
	
	xx = 720
	yy = 600
	
	__sousFen = c.rectangle(x1, y1, x2, y2)
	
	i = 3
	j = 1
	
	oi = -2  # (-2, -1) deviendra l'origine du repere
	oj = -1
	
	pi = 6  # dimensions du repere
	pj = 5
	
	i -= oi
	j -= oj
	
	# on rammene la taille a 1
	
	i /= pi
	j /= pj
	
	# mise a l'échelle (homothétie)
	
	i *= xx
	j *= yy
	
	# translation
	
	i += x1
	j += y1
	
	__rect1 = c.rectangle(i, j, i+1 ,j+1)
	
	print(i,"  ",j)
	
	c.assoc_key('z', bougerHaut)#attribution des touches zqsd pour le déplacement du mobile
	c.assoc_key('s', bougerBas)
	c.assoc_key('q', bougerGauche)
	c.assoc_key('d', bougerDroite)	
	
	
	
	c.main_loop()
	
	
main()













