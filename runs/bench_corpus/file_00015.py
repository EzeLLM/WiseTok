"""
a) Converta a temperatura de Celsius para Fahrenheit
(0 °C × 9/5) + 32 = 32 °F
b) Converta a Temperatura de Fahrenheit para Celsius
(32 °F − 32) × 5/9
c) Retorne verdadeiro se o número for PAR e falso se o número for IMPAR
d) faça um programa que peça as 4 notas bimestrais e mostre a media

e) faça um programa que receba um valor em KM, converta para: metros, centimetros e milimetros

f) faça um programa que pergunte quanto vc ganha por hora e o numero de horas trabalhadas no mês. Calcule e mostre
o total do seu salario no mes.

g) Tendo como dado de entrada a altura (h) de uma pessoa, construa um algoritmo que calcule seu peso ideal, utilizando
as seguintes formulas:
Para Homens (72.7*h)-58
Para Muleheres (62.1*h)44.7
"""


t = int(input("Digite a Temperatura para ser convertida para Fahrenheit: "))

print('A Temperatura em Fahrenheit é {}.'.format((t*9/5)+32))