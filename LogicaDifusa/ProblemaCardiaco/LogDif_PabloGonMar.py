#***********************************************************************************************************************
#                               PABLO GONZALEZ MARTIN - PRACTICA DE LOGICA DIFUSA
#                                  Computación Inteligente - 2018/2019 - MUSIANI
#***********************************************************************************************************************
# 
# 
#  Se va a crear un sistema de control difuso que modele la probabilidad de sufrir una enfermedad cardiáca. Se considerará
#  el rendimiento físico y la edad de los individuos, valorados entre 0 y 100, en función a los antecedentes anteriormente,
#  y la calidad alimentaria a la que se someten los sujetos, valorado entre 0 y 10. En función a los antecedentes
#  mencionados se obtendrá la probabilidad de sufrir una enfermedad cardiáca.
# 
# **********************************************************************************************************************
#                                           FORMULACIÓN DEL PROBLEMA
# **********************************************************************************************************************
#
# ANTECEDENTES (Entradas)
#     + Rendimiento físco: Universo: ¿Qué exigencia física tiene una persona, dentro de una escala de 0 a 100?
#     Conjunto difuso: bajo, medio, alto
# 
#     + Edad: Universo: ¿Qué edad tiene la persona a la que se destinará el estudio?
#     Conjunto difuso:  niño, adolescente, joven, adulto, viejo
#
#     + Alimentación: Universo: ¿Qué calidad alimentaria lleva el individuo al que se destina el estudio?
#     Conjunto difuso: mala, media, buena
#
# CONSECUENTES (Salidas)
#     + Problemas cardiácos: Universo: ¿Qué probabilidad hay de que se sufra una enfermedad cardiáca, en una escala de 0% a 100%?
#     Fuzzy set: baja, media, alta
# 
# **********************************************************************************************************************
#                                                       REGLAS
# **********************************************************************************************************************
#     1) Si el rendimiento físico es alto y la edad del individuo es adulto o viejo, entonces la probabilidad de padecer
# una enfermedad cardiaca es alta.
#     2) Si el rendimiento físico es bajo, entonces la probabilidad de padecer una enfermedad cardiaca es media.
#     3) Si el rendimiento físico es medio, entonces la probabilidad de padecer una enfermedad cardiaca es baja.
#     4) Si el rendimiento físico es alto y la edad del individuo no es viejo, entonces la probabilidad de padecer una
# enfermedad cardiaca es baja.
#     5) Si la edad del induviduo es viejo, entonces la probabilidad de padecer una enfermedad cardiaca es alta.
#     6) Si el rendimiento físico es medio y la edead del individuo es adulto o viejo, entonces la probabilidad de padecer
# una enfermedad cardiaca es baja.
#     7) Si la alimentacion es mala, entonces la probabilidad de padecer una enfermedad cardiaca es alta
#     8) Si la edad del individuo es adulto o viejo, y la alimentacion es mala la probabilidad de padecer una enfermedad
# cardiaca es alta.
#     10) Si el rendimiento físico es alto y la alimentacion es mala la probabilidad de padecer una enfermedad cardiaca es media.
#     11) Si el rendimiento físico es bajo, y la alimentacion es mala la probabilidad de padecer una enfermedad cardiaca es alta.
#     12) Si la alimentacion es buena, la probabilidad de padecer una enfermedad cardiaca es baja.
#     13) Si la alimentacion es media y el rendimiento fisico bajo, la probabilidad de padecer una enfermedad cardiaca es media.
#***********************************************************************************************************************

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# New Antecedent/Consequent objects hold universe variables and membership
# functions
performance = ctrl.Antecedent(np.arange(0, 101, 1), 'Rendimiento fisico')
age = ctrl.Antecedent(np.arange(0, 101, 1), 'Edad')
nutrition = ctrl.Antecedent(np.arange(0, 11, 1), 'Alimentacion')
heart_problem = ctrl.Consequent(np.arange(0, 101, 1), 'Problema cardiaco')


# Custom membership functions can be built interactively with a familiar,
# Pythonic API

#   ANTECEDENTE 'RENDIMIENTO'
performance['bajo'] = fuzz.trapmf(performance.universe, [0, 0, 20, 40])
performance['medio'] = fuzz.trapmf(performance.universe, [20, 40, 65, 80])
performance['alto'] = fuzz.trapmf(performance.universe, [60, 85, 100, 100])

#   ANTECENDETE 'EDAD'
age['ninio'] = fuzz.trapmf(age.universe, [0, 0, 10, 20])
age['joven'] = fuzz.trapmf(age.universe, [15, 20, 30, 40])
age['adulto'] = fuzz.trapmf(age.universe, [30, 40, 60, 70])
age['viejo'] = fuzz.trapmf(age.universe, [60, 75, 100, 100])

#   ANTECEDENTE 'ALIMENTACION'
nutrition['mala'] = fuzz.trimf(nutrition.universe, [0, 0, 4])
nutrition['media'] = fuzz.trimf(nutrition.universe, [2, 5, 8])
nutrition['buena'] = fuzz.trimf(nutrition.universe, [6, 10, 10])

#   CONSECUENTE 'PROBLEMA CARDIACO'
heart_problem['bajo'] = fuzz.trapmf(heart_problem.universe, [0, 0, 20, 40])
heart_problem['medio'] = fuzz.trapmf(heart_problem.universe, [20, 40, 60, 80])
heart_problem['alto'] = fuzz.trapmf(heart_problem.universe, [60, 80, 100, 100])

performance.view()
age.view()
nutrition.view()
heart_problem.view()

#***********************************************************************************************************************
#                                                   REGLAS BORROSAS
#***********************************************************************************************************************
#
#   Ahora, para hacer que estos antecedentes sean útiles, se define la relación difusa entre las variables de entrada y salida.
#   Para los propósitos de nuestro ejemplo, considere estas reglas simples:
# 
#     1) Si el rendimiento físico es alto y la edad del individuo es adulto o viejo, entonces la probabilidad de padecer
# una enfermedad cardiaca es alta.
#     2) Si el rendimiento físico es bajo, entonces la probabilidad de padecer una enfermedad cardiaca es media.
#     3) Si el rendimiento físico es medio, entonces la probabilidad de padecer una enfermedad cardiaca es baja.
#     4) Si el rendimiento físico es alto y la edad del individuo no es viejo, entonces la probabilidad de padecer una
# enfermedad cardiaca es baja.
#     5) Si la edad del induviduo es viejo, entonces la probabilidad de padecer una enfermedad cardiaca es alta.
#     6) Si el rendimiento físico es medio y la edead del individuo es adulto o viejo, entonces la probabilidad de padecer
# una enfermedad cardiaca es baja.
#     7) Si la alimentacion es mala, entonces la probabilidad de padecer una enfermedad cardiaca es alta
#     8) Si la edad del individuo es adulto o viejo, y la alimentacion es mala la probabilidad de padecer una enfermedad
# cardiaca es alta.
#     10) Si el rendimiento físico es alto y la alimentacion es mala la probabilidad de padecer una enfermedad cardiaca es media.
#     11) Si el rendimiento físico es bajo, y la alimentacion es mala la probabilidad de padecer una enfermedad cardiaca es alta.
#     12) Si la alimentacion es buena, la probabilidad de padecer una enfermedad cardiaca es baja.
#     13) Si la alimentacion es media y el rendimiento fisico bajo, la probabilidad de padecer una enfermedad cardiaca es media.
#  
# La mayoría de las personas estarían de acuerdo con estas reglas, aunque las reglas sean confusas. El reto es mapear
# reglas difusas en un cálculo preciso de la probabilidad de padecer una enfermedad cardiaca.

rule1 = ctrl.Rule(performance['alto'] & (age['adulto'] | age['viejo']), heart_problem['alto'])
rule2 = ctrl.Rule(performance['bajo'], heart_problem['medio'])
rule3 = ctrl.Rule(performance['medio'], heart_problem['bajo'])
rule4 = ctrl.Rule(age['viejo'], heart_problem['alto'])
rule5 = ctrl.Rule(performance['alto'] & (age['ninio'] | age['joven'] | age['adulto']), heart_problem['bajo'])
rule6 = ctrl.Rule(performance['medio'] & (age['adulto'] | age['viejo']), heart_problem['bajo'])
rule7 = ctrl.Rule(nutrition['mala'], heart_problem['alto'])
rule8 = ctrl.Rule((age['adulto'] | age['viejo']) & nutrition['mala'], heart_problem['alto'])
rule9 = ctrl.Rule(performance['alto'] & nutrition['mala'], heart_problem['medio'])
rule10 = ctrl.Rule(performance['bajo'] & nutrition['mala'] & age['viejo'], heart_problem['alto'])
rule11 = ctrl.Rule(performance['bajo'] & nutrition['media'] & (age['joven'] | age['adulto']), heart_problem['medio'])
rule12 = ctrl.Rule(nutrition['buena'], heart_problem['bajo'])
rule13 = ctrl.Rule(nutrition['media'] & performance['bajo'], heart_problem['medio'])

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Physical performance's values between 0-100
tipping.input['Rendimiento fisico'] = 30
# Age's values between 0-100
tipping.input['Edad'] = 38
# Nutrition's values between 0-10
tipping.input['Alimentacion'] = 7

# Crunch the numbers
tipping.compute()

print("La probabilidad de padecer una enfermedad cardiaca es:", tipping.output['Problema cardiaco'])
heart_problem.view(sim=tipping)
plt.show()

b = []

for d in range(0, 100):
    tipping.input['Rendimiento fisico'] = d
    tipping.compute()
    b.append(int(tipping.output['Problema cardiaco']))

plt.title('Fijado Edad y Alimentacion a: 30 años y 7 ptos.')
plt.ylabel("Probabilidad de enfermedad cardiaca (%)")
plt.xlabel("Rendimiento fisico (%)")
plt.plot(b)
plt.show()
