# Processo Seletivo – Intensivo Maker | Relatório do Candidato
### Théo Pereira de Souza
### https://github.com/theo-0413

## Resumo da Arquitetura do Modelo
A arquitetura utilizada seguiu as instruções sugeridas: depois da importação de todas as devidas bibliotecas, há o carregamento do dataset MNIST, junto do split explícito de treino/validação. A CNN conta com três blocos convolucionais, cada um deles contando com camadas de Conv2D, BatchNormalization, Ativação relu e MaxPooling2D respectivamente. Além do bloco de camadas densas, foi implementado um sistema de EarlyStopping, otimizando a eficiência da CNN uma vez que o modelo termina o treinamento se for detectado a parada do avanço da melhora.

## Bibliotecas Utilizadas
Dentre as bibliotecas utilizadas, detalhadas no requirements.txt, o projeto fez uso de:
- Módulo OS
- TensorFlow
- Keras 
- Scikit Learn 
- Numpy 

## Técnica de Otimização
Como sugerido, foi utilizada a técnica de otimização **Dynamic Range Quantization**, devido a praticidade da sua implementação e benefícios em termos de eficiência.

## Resultados Obtidos
Como esperado, o modelo mostra um valor de acurácia de validação consistente.

> Acurácia do teste: 0.9884 (Com perda de 0.038)

A otimização mostra os seguintes resultados:
> model.h5: 4362.2 KB
> 
> model.tflite: 367.4 KB

O que equivale a uma redução de tamanho de cerca de 91.56%. Por fim, a execução do run_inferece.py com 5 amostras retorna os seguintes resultados:
>Rodando inferencia em 5 amostras usando model.tflite:
> Amostra 1: predito=7 | real=7
>
> Amostra 2: predito=2 | real=2
>
> Amostra 3: predito=1 | real=1
>
> Amostra 4: predito=0 | real=0
>
> Amostra 5: predito=4 | real=4
>
Analisando essas amostras, é perceptível o quão preciso é o modelo construído a partir desta atividade prática.
