# lstm_brokers
Para que funcione hemos de pasarle por consola los campos necesarios con el siguiente formato:

python lstm_redirect_brokers.py "{'URL': '/', 'Session_ID':'david'}"

El formato es un diccionario en de cadena de texto. Este diccionario tendrá dos campos: 'URL', que es la url en la que se encuentra el usuario; y el 'Session_ID', que es el id de la sesión del usuario. Ambos campos se sacan del log.

El resultado será un número:

0
