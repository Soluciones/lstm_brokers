"""
Archivo utils_lstm_brokers.py
"""
# !pip install --upgrade boto3
import boto3
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

## Funciones de BBDD DynamoDB ##
def obtener_session_id(table):
  ids = []
  for dic in table.scan()['Items']:
    session_id = dic['Session_ID']
    ids.append(session_id)
  return ids

def anyadir_nueva_url(table, session_id, url, url2idx):
  response = table.get_item(Key={'Session_ID':session_id})
  item = response['Item']
  lista = item['URLs']
  url = limpiar_url_rankia(url)
  lista.append(url)
  lista_idx = [url2idx[url] for url in lista]
  return lista, lista_idx

def actualizar_tabla_nueva_url(table, session_id, url, url2idx):
  lista, lista_idx = anyadir_nueva_url(table, session_id, url, url2idx)
  table.update_item(Key={'Session_ID': session_id},
                    UpdateExpression='SET URLs = :val1',
                    ExpressionAttributeValues={
                    ':val1': lista,
                    })
  table.update_item(Key={'Session_ID': session_id},
                    UpdateExpression= 'SET IDXs = :val2',
                    ExpressionAttributeValues={
                    ':val2': lista_idx
                    })

def crear_nueva_entrada(table, session_id, url, url2idx):
  url = limpiar_url_rankia(url)
  idx = url2idx[url]
  table.put_item(
                Item={'Session_ID': session_id, 
                      'URLs': [url], 
                      'IDXs': [idx]}
                )

## Funciones de limpieza de URL ##
def limpiar_ver_contenido(url):
  if 'ver_contenido' in url or 'ver_padre' in url:
    return False
  else:
    return True

def limpiar_webinars(url):
  if 'anteriores' not in url:
    webi_list = url.split('/')
    if 'gracias' in webi_list:
      return '/' + webi_list[1] + '/' + webi_list[3]
    else:
      return '/' + webi_list[1] + '/webinar_cualquiera'
  else:
    return url

def limpiar_pixel_newsletter(url):
  if '/pixel/Newsletter' in url:
    return '/pixel/Newsletter'

def limpiar_home(url):
  url = '/' if 'home/' in url else url
  return url

def limpiar_url(url):
  url = url.split('?')[0]
  ids_posts_clave = ['2123190', '1527608', '4343391']
  if '/blog/' in url:
    url_blog = url.split('/')
    try:
      id_url = url_blog[3].split('-')[0]
    except:
      id_url = 'nada'
  else:
    id_url = 'nada'

  if id_url not in ids_posts_clave:
    if '/blog/' in url:
      url_ = url.split('/')
      url_final = '/' + url_[1] + '/' + url_[2]
    else:
      url_final = url
  else:
    url_final = url
  
  return url_final

def limpiar_url_rankia(url):
  url = limpiar_url(url)
  if 'webinars' in url:
    url = limpiar_webinars(url)
  elif '/pixel/Newsletter' in url:
    url = limpiar_pixel_newsletter(url)
  elif '/home/' in url:
    url = limpiar_home(url)
  else:
    url = url

  return url


### modelo
class BrokersRNN(nn.Module):
    """
    RNN para predecir qué sesión va a acabar contratando un webinar.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(BrokersRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sof = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sof_out = self.sof(out)
        sof_out = sof_out.view(batch_size, -1, self.output_size)
        sof_out = sof_out[:, -1]    
        return sof_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

## Función final
def convierte(dict_, url2idx, ann, table):
  # dict_ = ast.literal_eval(dict_string)
  session_id = dict_['Session_ID']
  url = limpiar_url_rankia(dict_['URL'])
  lista_session_ids = obtener_session_id(table)
  if session_id in lista_session_ids:
    actualizar_tabla_nueva_url(table, session_id, url, url2idx)
    response = table.get_item(Key={'Session_ID':session_id})
    item = response['Item']
    lista = item['IDXs']
    if len(lista) >175:
      lista = lista[-175:]
    else:
      lista_0 = [0]*(175-len(lista))
      lista = lista_0 + lista
    lista = torch.FloatTensor(lista)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lista = lista.to(device)
    train_on_gpu=torch.cuda.is_available()
    h = ann.init_hidden(1)
    h = tuple([each.data for each in h])
    prediccion, h = ann(lista.unsqueeze(0), h)
    prediccion = int(torch.argmax(prediccion))
  else:
    crear_nueva_entrada(table, session_id, url, url2idx)
    prediccion = 0
  return prediccion
