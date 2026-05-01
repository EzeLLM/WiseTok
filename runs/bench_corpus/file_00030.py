
mensaje ="Archivo 'Tigo201903_1.txt' de tipo 'text/plain' totalmente cargado! con id de proceso: 1578512783614 ";

mensaje_sin_espacios = mensaje.split(":")

id_proceso = mensaje_sin_espacios[1].lstrip().rstrip()

print(mensaje)
print(id_proceso)