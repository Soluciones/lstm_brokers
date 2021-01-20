import boto3
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json
import os
from bottle import post, request, run

from utils_lstm_brokers import obtener_session_id
from utils_lstm_brokers import anyadir_nueva_url
from utils_lstm_brokers import actualizar_tabla_nueva_url
from utils_lstm_brokers import crear_nueva_entrada
from utils_lstm_brokers import limpiar_ver_contenido
from utils_lstm_brokers import limpiar_webinars
from utils_lstm_brokers import limpiar_pixel_newsletter
from utils_lstm_brokers import limpiar_home
from utils_lstm_brokers import limpiar_url
from utils_lstm_brokers import limpiar_url_rankia
from utils_lstm_brokers import BrokersRNN
from utils_lstm_brokers import convierte


aws_access_key_id = os.environ['ACCESS_KEY']
aws_secret_access_key = os.environ['SECRET_ACCESS_KEY']
region = os.environ['REGION']

table_name = 'Prueba_David_1'

dynamodb = boto3.resource('dynamodb', region_name = region, 
                          aws_access_key_id = aws_access_key_id, 
                          aws_secret_access_key = aws_secret_access_key)

table = dynamodb.Table(table_name)

#Cargamos modelo
with open('url2idx.txt', 'r') as file:
  for line in file:
    url2idx = json.loads(line)
file.close()

vocab_size = len(url2idx) + 1
output_size = 15
embedding_dim = 300
hidden_dim = 256
n_layers = 3

net = BrokersRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.load_state_dict(torch.load('model1.pt', map_location=device))

@post('/rnn_brokers')
def prediccion():
  data = request.body.getvalue().decode('utf-8')
  dic_input = ast.literal_eval(data)
  resultado = convierte(dic_input, url2idx, net, table)
  return str(resultado)

run(host='0.0.0.0', port=8787)
