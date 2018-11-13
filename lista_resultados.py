# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

def check_is_dir(path):
    import os
    print('Checando: ', path)
    if not os.path.isdir(path):
        print('Criando: ', path)
        os.makedirs(path)
        
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

'''
def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result
'''

def read_tensors_from_image_files(file_names, input_height=299, input_width=299, input_mean=0, input_std=255):
   with tf.Graph().as_default():
    input_name = "file_reader"
    output_name = "normalized"
    file_name_placeholder = tf.placeholder(tf.string, shape=[])
    file_reader = tf.read_file(file_name_placeholder, input_name)
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
        name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    with tf.Session() as sess:
      for file_name in file_names:
          if file_name.endswith('.jpg'):
            yield file_name, sess.run(normalized, {file_name_placeholder: file_name})

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.ERROR)
  file_name = "0icmiIy2Fd4BDPgc.jpg"
  model_file = "output_graph.pb"
  label_file = "output_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"
  validation_dir = 'k-fold/validacao/NAO_OK/TECNICO/IMAGEM_ESCURA/'

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--validation_dir", help="name of validation dir")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  if args.validation_dir:
    validation_dir = args.validation_dir

  graph = load_graph(model_file)
  
  scores = {}
  
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  sess = tf.Session(graph=graph)

  labels = load_labels(label_file)

  i = 0
  import os
  import shutil
  print('comecando a listar imagems')
  '''
  for lbl in labels:
      check_is_dir(os.path.join('predicoes', 'caminhao', '_'.join(lbl.split(' '))))
  '''
  arq = open('percents.txt', 'w')

  for r, d, files in os.walk(validation_dir):
      r = os.path.normpath(r)
      classe_pred = r.split('/')[-1]
      imgs = []
      nome_img = {}
      file_names = []
      for file in files:
          file_name = os.path.join(r,file)
          file_names.append(file_name)

      for nome, img in read_tensors_from_image_files(
                  file_names,
                  input_height=input_height,
                  input_width=input_width,
                  input_mean=input_mean,
                  input_std=input_std):
        nome_img[nome] = img
        # imgs.append(img)
        i += 1
        print(i)
    
      print('Acabou de listar imagens.')
      print('Predicting in: ', classe_pred)
    
      i = 0
      # check_is_dir(os.path.join('predicoes', classe_pred))
      for nome, img in nome_img.items():
        results = sess.run(output_operation.outputs[0], {
                           input_operation.outputs[0]: img})
        
        results = np.squeeze(results)
             
        top_k = results.argsort()[-5:][::-1]

        preded = '_'.join(labels[top_k[0]].split(' '))
        true_pred = classe_pred+'--'+preded
        # true_pred = (r.split('/')[-1])+'--'+'_'.join(preded.split(' '))
        # true_pred = ' '.join((r.split('/')[-1]).split('_')[-1])+'--'+preded
        if true_pred in scores:
          scores[true_pred] += 1
        else:
          scores[true_pred] = 1 
        i+=1
        print(i, classe_pred + ': ' + preded)

        arq.write('Image name: ' + nome + '\n')
        arq.write('Prediction: ' + preded + '\n')
        for aux in top_k:
            # print(labels[aux],':',results[aux])
            arq.write(str(labels[aux]) + ':' + str(results[aux]) + '\n')
        arq.write('----------------------------\n')
        # copia arq original para pasta de predicoes/true_pred/preded 
        # para otimizar, tirar do loop de predicao e fazer um dicionario que guarda o nome
        # dos arquivos, e chamar a copia num loop separado
        # check_is_dir(os.path.join('predicoes', classe_pred, preded))
        # shutil.copy(nome, os.path.join('predicoes', 'caminhao', preded, nome.split('/')[-1])) 

   
  print(scores)
  arq.close()
  
  import json
  json = json.dumps(scores)
  with open('preds_dic.json', 'w') as f:
      f.write(json)      
  from scorings import accuracy, recall, precision, F1_score
  print('acuracia modelo: ', accuracy(scores))
  
  for nome in labels:
      rec = recall(scores, '_'.join(nome.split(' ')))
      pre = precision(scores, '_'.join(nome.split(' ')))
      print('\nrecall '+nome+':', rec)
      print('precision '+nome+':', pre)
      print('F1_score '+nome+':', F1_score(rec, pre))
  

    
  
  # print('ok: ', acertos['ok_carro'], '\nescuro: ', acertos['escuro'])
          
        
