import mysql.connector


def getConnexion():

  db = mysql.connector.connect(
	host = "localhost",
	user = "root",
	password = "",
	database="uvprojet1"
	)


  print(db)

  return db	

# sa va afficher quelque chose comme un emplacement menoire c'est a dire en hexadecimal et la on peut maintenant interroger nos base de donne